# -*- coding: utf-8 -*-
"""Lo que la integración del Paso 2 añadió encima de las piezas (26-sep-2026).

Cada comprobación nombra el fallo que cierra (las 5 a 9, de la revisión
adversarial de la integración, 26-sep-2026):
  · la propuesta global vivía en la memoria de un worker y el árbol la leía
    distinta según quién contestara (revisión adversarial de la pantalla);
  · la cita de `presupone` que es el planteamiento entero «constaba» siempre
    (medición de la fase 5: las dos citas del motor eran el párrafo entero);
  · emparejar conservaba la pregunta reformulada por el modelo y el árbol
    perdía el problema;
  · el asunto podía prosperar por un accesorio que nadie calificó sin que el
    secretario lo leyera (6 de 16 engroses con el principal desestimado).
"""
import ast
import asyncio
import copy
import dataclasses
import json
import os
import time
import types

os.environ.pop("ESTUDIO_PROMPT", None)
os.environ.pop("PLAN_ESTUDIO", None)

from fastapi import HTTPException

import arbol_decision as ad
import fase5_propuesta as f5
import fase6_estudio as f6
import fases123_pipeline as f123
import recalificar as rc
import taller_estado as _te

FALLAS = []


def ok(cond, msg):
    print(("   PASA   " if cond else "   FALLA  ") + msg)
    if not cond:
        FALLAS.append(msg)


SRC_MAIN = open("main.py", encoding="utf-8").read()
ARBOL = ast.parse(SRC_MAIN)
FN = {n.name: n for n in ast.walk(ARBOL) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}

print("1 · LA PROPUESTA GLOBAL, EN LA FILA Y LEÍDA IGUAL POR LAS CINCO PUERTAS")
_src_rep = ast.get_source_segment(SRC_MAIN, FN["taller_reparto"])
ok("_taller_glob(global_json, ses)" in _src_rep,
   "/taller/reparto lee la global como las demás puertas (la del cliente o la guardada)")
_src_rec = ast.get_source_segment(SRC_MAIN, FN["_taller_recuperar_sesion"])
ok("_taller_global_de_fila(est, resultado)" in _src_rec,
   "la sesión rehidratada de la base trae la global")
_src_prop = ast.get_source_segment(SRC_MAIN, FN["taller_proponer"])
ok(_src_prop.count("_taller_guardar_global(") == 2,
   "/taller/proponer la guarda al servir la calculada sola y al calcularla con contexto")
# La función que la lee, sin la base: se ejecuta su fuente con dobles.
_ns = {"_te": types.SimpleNamespace(huella_contraste=lambda r: r["h"]), "_types": types}
exec(compile(ast.Module(body=[FN["_taller_global_de_fila"]], type_ignores=[]), "main.py", "exec"), _ns)
_leer = _ns["_taller_global_de_fila"]
_G = {"sentido": "infundado", "checklist": [{"tema": "1"}], "alcanza": False}
ok(_leer({"global_propuesta": {"huella": "A", "global": _G}}, {"h": "A"}).checklist == [{"tema": "1"}],
   "la marca propia, con la huella del adelanto")
ok(_leer({"global_propuesta": {"huella": "VIEJA", "global": _G}}, {"h": "A"}) is None,
   "la de otro adelanto no vale")
ok(_leer({"propuesta": {"huella": "A", "respuesta": {"global": dict(_G, sentido="fundado")}}},
         {"h": "A"}).sentido == "fundado",
   "sin marca propia, la de la propuesta calculada sola")
ok(_leer({"global_propuesta": {"huella": "A", "global": _G},
          "propuesta": {"huella": "A", "respuesta": {"global": dict(_G, sentido="fundado")}}},
         {"h": "A"}).sentido == "infundado",
   "la última que vio la pantalla manda sobre la calculada sola")
ok(_leer({}, {"h": "A"}) is None, "sin nada guardado, nada")

print("\n2 · LA CITA DE «PRESUPONE» ES UN PASAJE, NO EL PLANTEAMIENTO ENTERO")
_comb = ("la sala debió estudiar los alegatos contra el crédito porque el crédito formaba parte "
         "de la litis desde la demanda y la responsable lo reconoció al admitir la ampliación")
_acc = {"pregunta": "¿Debió estudiar los alegatos?", "combate": _comb}
_pral = {"pregunta": "¿El crédito formaba parte de la litis?", "combate": "el crédito formaba parte de la litis"}
_e_todo = {"presupone": {"premisa": "el crédito en la litis", "cita": _comb, "causa_propia": None}}
ok(ad.presupuesto(_e_todo, _acc, _pral)["motivo"] == "cita_extensa",
   "el planteamiento entero no prueba dónde da por cierta la premisa")
_e_pas = {"presupone": {"premisa": "el crédito en la litis",
                        "cita": "porque el crédito formaba parte de la litis", "causa_propia": None}}
ok(ad.presupuesto(_e_pas, _acc, _pral)["verificado"],
   "el pasaje donde la da por cierta, sí")

print("\n3 · EMPAREJAR DEVUELVE LA PREGUNTA DE LA FASE 3")
_p_mod = f5.Propuesta(problema="¿Procede la condena en costas?", sentido="infundado")
_fase3 = [{"pregunta": "¿Es legal el crédito?"}, {"pregunta": "¿Procede la condena en costas impuesta en la sentencia?"}]
_emp = f5.emparejar(_fase3, [_p_mod])
ok(_emp[0] is None and _emp[1] is not None
   and _emp[1].problema == "¿Procede la condena en costas impuesta en la sentencia?",
   "emparejada por tema, lleva el texto de la fase 3, que es el que busca el árbol")
ok(_p_mod.problema == "¿Procede la condena en costas?", "y el objeto del modelo no se toca")

print("\n4 · EL ASUNTO QUE PROSPERA POR UN ACCESORIO SE DICE")
P1, P2, P3 = "¿Es procedente la acción?", "¿Hubo incongruencia en la condena?", "¿Proceden las costas?"
_probs = [{"pregunta": P1, "jerarquia": "principal"}, {"pregunta": P2}, {"pregunta": P3}]


def _crit(s2, t2=False):
    return [{"problema": P1, "sentido": "infundado", "jerarquia": "principal", "tocado": True},
            {"problema": P2, "sentido": s2, "razonamiento": "r", "jerarquia": "accesorio", "tocado": t2},
            {"problema": P3, "sentido": "infundado", "razonamiento": "r", "jerarquia": "accesorio"}]


_av, _ = ad.aplicar(_probs, _crit("fundado"), [], [], sentido_motor="infundado")
ok(any("EL ASUNTO PROSPERARÍA" in a and P2[:40] in a for a in _av),
   "un accesorio «fundado» que él no tocó, con el principal desestimado: se avisa")
_av2, _ = ad.aplicar(_probs, _crit("fundado", t2=True), [], [], sentido_motor="infundado")
ok(not any("EL ASUNTO PROSPERARÍA" in a for a in _av2), "si lo marcó él, es su decisión: no se avisa")
_av3, _ = ad.aplicar(_probs, _crit("infundado"), [], [], sentido_motor="infundado")
ok(not any("EL ASUNTO PROSPERARÍA" in a for a in _av3), "si nada prospera, nada que avisar")


# ═══ LO QUE SIGUE: LA REVISIÓN ADVERSARIAL DE LA INTEGRACIÓN (26-sep-2026) ═══
# Las funciones reales de main.py, sacadas por AST y ejecutadas con dobles de
# la base y del modelo: el camino de verdad de las puertas, no una copia.
_NOMBRES = ("_taller_tocados", "_taller_glob", "_taller_armar_criterio", "_taller_plan_cas",
            "_taller_plan_leer", "_taller_recalificadas_guardadas", "_taller_recalificar_correr",
            "_taller_recalificar_lanzar", "_taller_recalificar_para",
            "_taller_recalificado_al_resolver", "_taller_avisos_foto", "_taller_avisos_restaurar",
            "_taller_criterios_pantalla", "_taller_recalificado_al_pedir")


def entorno(**extra):
    ns = {"json": json, "os": os, "time": time, "asyncio": asyncio, "err": lambda e: str(e),
          "HTTPException": HTTPException, "_te": _te, "supabase_admin": None,
          "chat_client": object(), "_TALLER_EN_MARCHA": set(), "print": lambda *a, **k: None,
          "_types": types}
    ns.update(extra)
    exec(compile(ast.Module(body=[FN[n] for n in _NOMBRES if n in FN], type_ignores=[]), "main.py",
                 "exec"), ns)
    return ns


NS = entorno()


def resultado(problemas, tipo="amparo_directo"):
    r = types.SimpleNamespace(fases=f123.Fases123(resumen_acto="La Sala resolvió.",
                                                  problemas=copy.deepcopy(problemas),
                                                  fuentes=["", ""]))
    r.fases.avisos = []
    r.encargo = types.SimpleNamespace(tipo_asunto=tipo, es_recurso=False, variante_estudio="v1",
                                      formato="", suplencia={}, conceptos_violacion="",
                                      propuesta_global={}, plan={}, guion="", resolvio_declarado="",
                                      responsable="")
    r.avisos = []
    return r


def las_dos_puertas(problemas, props, glob, cj):
    """Lo que decide la pantalla (/taller/reparto, texto entero) y lo que deciden
    los gemelos (`_taller_armar_criterio`, recortado a 400), por la clave común."""
    rep = ad.reparto_para_pantalla(copy.deepcopy(problemas), copy.deepcopy(cj), glob.get("checklist") or [],
                                   copy.deepcopy(props), sentido_motor=glob.get("sentido", ""),
                                   tipo_asunto="amparo_directo", huella_adelanto="H")
    pant = {ad.clave_problema(c["problema"]): (c["sentido"], c["razonamiento"], c["de"] or "",
                                               bool(c["recalificar"])) for c in rep["criterios"]}
    r = resultado(problemas)
    ses = {"propuestas": [types.SimpleNamespace(**p) for p in props], "material": None}
    arm = NS["_taller_armar_criterio"](r, ses, glob, criterios_json=json.dumps(cj, ensure_ascii=False))
    gem = {ad.clave_problema(c.problema): (c.sentido, c.razonamiento,
                                           (arm["detalle"].get(c.problema) or {}).get("de", "") or "",
                                           bool((arm["detalle"].get(c.problema) or {}).get("recalificar")))
           for c in arm["crit"]}
    return pant, gem, rep["avisos"], arm


print("\n5 · LOS PLANTEAMIENTOS DE MÁS DE 400 CARACTERES DECIDEN IGUAL EN LA PANTALLA Y EN LOS GEMELOS")
L_ACC = ("¿La condena en costas se sustentó en la conducta procesal de la cedente? "
         + "Se insiste en que la condena en costas no atendió a la conducta procesal de las partes. " * 5)
L_PRAL = ("¿La cesión del contrato de arrendamiento celebrada entre la cedente y la cesionaria, "
          "notificada a la arrendadora mediante escrito presentado ante notario y aceptada tácitamente "
          "por ella al recibir los pagos de la cesionaria durante los meses de enero a junio, liberó "
          "a la cedente de responder por las rentas reclamadas en el juicio de origen, conforme a los "
          "artículos del Código Civil que regulan la cesión de deudas y la novación subjetiva, y con "
          "independencia de la cláusula de solidaridad pactada en el contrato original?")
C1 = "¿La cesión del contrato liberó a la cedente de responder por las rentas reclamadas?"
C3 = "¿La Sala debió pronunciarse sobre la excepción de pago parcial opuesta en la contestación?"
assert len(L_ACC) > 400 and len(L_PRAL) > 400
MI_RAZON = "MI RAZÓN: la pericial contable no se valoró"


def _prop(p, s, razon="r", alcanza=True):
    return {"problema": p, "sentido": s, "razon": razon, "alcanza": alcanza,
            "sentido_propio": "", "razon_propia": ""}


# (a) Accesorio largo que él marcó, CON cambio de sentido del principal.
F3a = [{"pregunta": C1, "jerarquia": "principal", "combate": "la cesión liberó"},
       {"pregunta": L_ACC, "jerarquia": "accesorio", "depende_de": 1, "combate": "las costas no proceden"},
       {"pregunta": C3, "jerarquia": "accesorio", "combate": "omitió la excepción"}]
PRa = [_prop(C1, "fundado"), _prop(L_ACC, "fundado"), _prop(C3, "infundado")]
CJa = [{"problema": C1, "sentido": "infundado", "razonamiento": "la cesión no se notificó",
        "jerarquia": "principal", "tocado": True},
       {"problema": L_ACC, "sentido": "fundado", "razonamiento": MI_RAZON, "jerarquia": "accesorio",
        "tocado": True},
       {"problema": C3, "sentido": "infundado", "razonamiento": "los recibos no acreditan",
        "jerarquia": "accesorio", "tocado": False}]
pant, gem, _, arm_a = las_dos_puertas(F3a, PRa, {"sentido": "fundado", "alcanza": True, "checklist": []}, CJa)
_kL = ad.clave_problema(L_ACC)
ok(pant == gem, "(a) con cambio de sentido, la pantalla y los gemelos deciden igual")
ok(gem[_kL] == ("fundado", MI_RAZON, "tuya", False) and not rc.pendientes(arm_a["detalle"]),
   "(a) el accesorio largo que él marcó sigue siendo SUYO: ni se tumba ni lo recalifica el motor")
ok(NS["_taller_criterios_pantalla"](arm_a)[1]["tocado"],
   "(a) /taller/recalificar lo devuelve «tocado», como la pantalla")
# (b) Accesorio largo que él marcó, SIN cambio de sentido (el principal, como el motor).
CJb = copy.deepcopy(CJa)
CJb[0]["sentido"], CJb[1]["sentido"] = "fundado", "infundado"
pant, gem, _, arm_b = las_dos_puertas(F3a, PRa, {"sentido": "fundado", "alcanza": True, "checklist": []}, CJb)
ok(pant == gem and gem[_kL] == ("infundado", MI_RAZON, "tuya", False),
   "(b) sin cambio de sentido, su marca manda en las dos puertas (no pasa a «innecesario»)")
# (c) Principal largo que él acepta tal cual: nada se recalifica.
F3c = [{"pregunta": L_PRAL, "jerarquia": "principal", "combate": "la cesión liberó"},
       {"pregunta": C1 + " (costas)", "jerarquia": "accesorio", "depende_de": 1, "combate": "costas"},
       {"pregunta": C3, "jerarquia": "accesorio", "combate": "omitió la excepción"}]
PRc = [_prop(L_PRAL, "infundado"), _prop(C1 + " (costas)", "infundado"), _prop(C3, "fundado")]
CJc = [{"problema": L_PRAL, "sentido": "infundado", "razonamiento": "r", "jerarquia": "principal",
        "tocado": False},
       {"problema": C1 + " (costas)", "sentido": "infundado", "razonamiento": "r", "jerarquia": "accesorio",
        "tocado": False},
       {"problema": C3, "sentido": "fundado", "razonamiento": "r", "jerarquia": "accesorio", "tocado": False}]
pant, gem, _, arm_c = las_dos_puertas(F3c, PRc, {"sentido": "fundado", "alcanza": True, "checklist": []}, CJc)
ok(pant == gem and not rc.pendientes(arm_c["detalle"])
   and not any("SE RECALIFICAN" in a for a in arm_c["avisos_fases"]),
   "(c) principal largo aceptado tal cual (global fundado): ningún tumbado en ninguna puerta")
# (d) Principal largo fundado que NO alcanza: ninguna puerta aplica la sustracción.
PRd = [_prop(L_PRAL, "fundado", alcanza=False), _prop(C1 + " (costas)", "infundado"), _prop(C3, "infundado")]
CJd = copy.deepcopy(CJc)
CJd[0]["sentido"], CJd[2]["sentido"] = "fundado", "infundado"
pant, gem, av_p, arm_d = las_dos_puertas(F3c, PRd, {"sentido": "fundado", "alcanza": False, "checklist": []},
                                         CJd)
ok(pant == gem and gem[ad.clave_problema(C1 + " (costas)")][0] == "infundado"
   and any("NO SE APLICÓ LA SUSTRACCIÓN" in a for a in av_p)
   and any("NO SE APLICÓ LA SUSTRACCIÓN" in a for a in arm_d["avisos_fases"]),
   "(d) principal largo con alcanza=False: las dos puertas lo ven y no aplican la sustracción")
ok(ad.clave_problema(L_PRAL) == L_PRAL[:ad.CORTE_PROBLEMA] and ad.clave_problema(None) == "",
   "la clave es una sola: el recorte del criterio armado")


print("\n6 · CON ACCESORIOS SIN CALIFICAR, LOS DOS GEMELOS NO GENERAN (Y NO COBRAN)")
# El caso: el motor propuso el principal fundado y los accesorios sin materia;
# el secretario lo resuelve infundado y el árbol tumba los dos accesorios. El
# modelo de la recalificación falla (un 503): quedan pendientes.
P1g = "¿Es procedente la vía ejecutiva mercantil ejercida por la actora?"
P2g = "¿La Sala debió estudiar los agravios sobre los intereses moratorios pactados?"
P3g = "¿Procede la condena en costas de segunda instancia?"
F3g = [{"pregunta": P1g, "jerarquia": "principal", "combate": "el pagaré no es título ejecutivo"},
       {"pregunta": P2g, "jerarquia": "accesorio", "depende_de": 1, "combate": "omitió la tasa"},
       {"pregunta": P3g, "jerarquia": "accesorio", "depende_de": 1, "combate": "no hubo temeridad"}]
PRg = [_prop(P1g, "fundado"), _prop(P2g, "innecesario"), _prop(P3g, "innecesario")]
CJg = json.dumps([
    {"problema": P1g, "sentido": "infundado", "razonamiento": "el pagaré sí es título ejecutivo",
     "jerarquia": "principal", "tocado": True},
    {"problema": P2g, "sentido": "innecesario", "razonamiento": "sin materia", "jerarquia": "accesorio",
     "tocado": False},
    {"problema": P3g, "sentido": "innecesario", "razonamiento": "sin materia", "jerarquia": "accesorio",
     "tocado": False}], ensure_ascii=False)
GLOBg = {"sentido": "fundado", "alcanza": True, "checklist": []}
LLAMADAS = {"plan": 0, "resolver": 0, "en_vivo": 0, "uso": 0, "cobro": 0, "puerta": []}


async def _llamar_mal(cliente, texto):
    raise RuntimeError("503")


async def _llamar_bien(cliente, texto):
    import re as _re_t
    nums = [int(x) for x in _re_t.findall(r"PLANTEAMIENTO (\d+) ·", texto)]
    return json.dumps({"planteamientos": [
        {"numero": n, "sentido": "infundado",
         "razon": f"con la premisa, el planteamiento {n} no tiene razón por sus propios méritos",
         "presupone": None} for n in nums]})


async def _plan_espia(*a, **k):
    LLAMADAS["plan"] += 1
    return {"estado": "no_aplica"}


class _Parar(Exception):
    pass


async def _resolver_espia(*a, **k):
    LLAMADAS["resolver"] += 1
    raise _Parar()


async def _en_vivo_espia(*a, **k):
    LLAMADAS["en_vivo"] += 1
    yield {"tipo": "componiendo"}


async def _parametro(*a, **k):
    return "", None


def _gemelos_ns():
    ns = entorno()
    ns.update({
        "Form": lambda *a, **k: "", "_taller_puerta": lambda u, cobrable=False: LLAMADAS["puerta"].append(cobrable),
        "_taller_purgar": lambda: None, "_decidir_oportunidad": lambda *a, **k: None,
        "_taller_variante_estudio": lambda *a, **k: "v1", "_taller_parametro": _parametro,
        "_taller_direccion_al_material": lambda *a, **k: None, "_con_autos": lambda r, c: c or "",
        "_taller_plan_para": _plan_espia, "_taller_plan_aplica": lambda r: False,
        "_taller_registrar_uso": lambda *a, **k: LLAMADAS.__setitem__("uso", LLAMADAS["uso"] + 1),
        "_taller_cobrar": lambda *a, **k: LLAMADAS.__setitem__("cobro", LLAMADAS["cobro"] + 1),
        "_TALLER_LATIDO_S": 0.2, "qdrant_client": None})
    for n in ("taller_resolver", "taller_resolver_stream"):
        nodo = copy.deepcopy(FN[n])
        nodo.decorator_list = []
        exec(compile(ast.Module(body=[nodo], type_ignores=[]), "main.py", "exec"), ns)
    return ns


def _sesion():
    r = resultado(F3g)
    r.avisos = ["UN AVISO DEL ADELANTO"]
    r.fases.avisos = ["UN AVISO DE LAS FASES"]
    ses = {"resultado": r, "propuestas": [types.SimpleNamespace(**p) for p in PRg],
           "material": types.SimpleNamespace(), "tmp": "/tmp"}
    return ses, r


_FORMg = dict(numero="1/2026", user_email="x@y.mx", sentido="", problema="", razonamiento="",
              criterios_json=CJg, contexto="", usar_propuesta=False, modo_decision="", sentido_global="",
              global_dictado="", resolvio_declarado="", global_json=json.dumps(GLOBg), conceptos_violacion="",
              responsable="", oportunidad_decision="", oportunidad_motivo="", formato="", variante_estudio="",
              suplencia="", razones_segmento="")


async def _flujo(ns):
    resp = await ns["taller_resolver_stream"](**_FORMg)
    evs = []
    async for trozo in resp.body_iterator:
        t = trozo.decode() if isinstance(trozo, bytes) else trozo
        if t.startswith("data: "):
            evs.append(json.loads(t[6:]))
    return evs


import redactor_adelanto as _ra_t  # noqa: E402
_orig = (rc._llamar, _ra_t.resolver, _ra_t.resolver_en_vivo)
_ra_t.resolver, _ra_t.resolver_en_vivo = _resolver_espia, _en_vivo_espia
try:
    rc._llamar = _llamar_mal
    GN = _gemelos_ns()
    ses_f, r_f = _sesion()
    GN["_taller_recuperar_sesion"] = lambda u, n: ses_f
    evs = asyncio.run(_flujo(GN))
    _err = [e for e in evs if e.get("tipo") == "error"]
    ses_p, r_p = _sesion()
    GN["_taller_recuperar_sesion"] = lambda u, n: ses_p
    try:
        asyncio.run(GN["taller_resolver"](**_FORMg))
        _409 = None
    except HTTPException as ex:
        _409 = ex
    except _Parar:                 # llegó al estudio: generó con los sin calificar
        _409 = None
    ok(len(_err) == 1 and "SIN CALIFICAR TRAS TU CAMBIO DE SENTIDO" in _err[0]["mensaje"]
       and P2g[:60] in _err[0]["mensaje"] and P3g[:40] in _err[0]["mensaje"]
       and "Califícalos tú en la pantalla" in _err[0]["mensaje"]
       and "vuelve a generar para reintentar" in _err[0]["mensaje"],
       "flujo: evento «error» que nombra los planteamientos y dice qué hacer")
    ok(_409 is not None and _409.status_code == 409 and _err and _409.detail == _err[0]["mensaje"],
       "plano: 409 con el MISMO mensaje (los dos gemelos deciden igual)")
    ok(LLAMADAS["plan"] == 0 and LLAMADAS["resolver"] == 0 and LLAMADAS["en_vivo"] == 0,
       "ninguno llega al plan ni al estudio: el bloque «SIN CALIFICAR» de la v1 no se alcanza")
    ok(LLAMADAS["uso"] == 0 and LLAMADAS["cobro"] == 0 and LLAMADAS["puerta"] == [True, True],
       "no se registra el uso ni se cobra (la puerta cobrable sólo mira la bolsa)")
    ok(r_f.avisos == ["UN AVISO DEL ADELANTO"] and r_f.fases.avisos == ["UN AVISO DE LAS FASES"]
       and r_p.avisos == ["UN AVISO DEL ADELANTO"] and r_p.fases.avisos == ["UN AVISO DE LAS FASES"],
       "los avisos de la petición cortada («SE RECALIFICAN…») no se quedan en el adelanto")
    # Control: con la recalificación que llega, los dos siguen al plan y al estudio.
    rc._llamar = _llamar_bien
    ses_f2, _ = _sesion()
    GN["_taller_recuperar_sesion"] = lambda u, n: ses_f2
    evs2 = asyncio.run(_flujo(GN))
    ses_p2, _ = _sesion()
    GN["_taller_recuperar_sesion"] = lambda u, n: ses_p2
    try:
        asyncio.run(GN["taller_resolver"](**_FORMg))
    except _Parar:
        pass
    ok(not [e for e in evs2 if e.get("tipo") == "error"] and LLAMADAS["plan"] == 2
       and LLAMADAS["en_vivo"] == 1 and LLAMADAS["resolver"] == 1,
       "control: con la recalificación hecha, los dos gemelos siguen al plan y al estudio")
finally:
    rc._llamar, _ra_t.resolver, _ra_t.resolver_en_vivo = _orig
_m_no = rc.aviso_sin_calificar([P2g], "infundado", reintentable=False, motivo="tope")
ok("vuelve a generar" not in _m_no and "Califícalos tú en la pantalla" in _m_no and "se agotaron" in _m_no,
   "si volver a generar no lo reintentaría (corridas agotadas), no se le dice que vuelva a generar")
print()
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN" if not FALLAS else f"FALLAN {len(FALLAS)}: " + " · ".join(FALLAS))
raise SystemExit(1 if FALLAS else 0)
