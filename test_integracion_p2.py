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
    LLAMADAS.setdefault("cv", []).append(a[1].encargo.conceptos_violacion)
    raise _Parar()


async def _en_vivo_espia(*a, **k):
    LLAMADAS["en_vivo"] += 1
    LLAMADAS.setdefault("cv", []).append(a[1].encargo.conceptos_violacion)
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


print("\n7 · LA RECALIFICACIÓN NO DECLARA «INNECESARIO» LO QUE EL ÁRBOL MANDA ESTUDIAR")
# El caso de la revisión: el motor propuso el principal infundado y el
# secretario lo resuelve FUNDADO; el accesorio es la prescripción (pide más que
# lo concedido: art. 189). En las dos ramas del árbol que lo mandan estudiar.
P1p = "¿La cesión del contrato liberó a la cedente de responder por las rentas reclamadas?"
P5p = "¿Operó la prescripción de la acción de cobro de las rentas reclamadas a la cedente?"
P6p = "¿La condena en costas se sustentó en la conducta procesal de la cedente?"
F3p = [{"pregunta": P1p, "jerarquia": "principal", "combate": "la cesión liberó a la cedente"},
       {"pregunta": P5p, "jerarquia": "accesorio", "depende_de": 1, "combate": "la acción prescribió"},
       {"pregunta": P6p, "jerarquia": "accesorio", "depende_de": 1, "combate": "las costas no proceden"}]


def _crit_p():
    return [{"problema": P1p, "sentido": "fundado", "razonamiento": "la cesión se notificó",
             "jerarquia": "principal", "tocado": True},
            {"problema": P5p, "sentido": "infundado", "razonamiento": "r", "jerarquia": "accesorio"},
            {"problema": P6p, "sentido": "infundado", "razonamiento": "r", "jerarquia": "accesorio"}]


for _rama, _alc, _quien, _motivo in (("pide más (art. 189)", True, P5p, "mayor_beneficio"),
                                     ("lo fundado no alcanza", False, P6p, "no_alcanza")):
    _props_p = [_prop(P1p, "infundado", alcanza=_alc), _prop(P5p, "infundado"), _prop(P6p, "infundado")]
    _cr = _crit_p()
    _av, _det = ad.aplicar(copy.deepcopy(F3p), _cr, [], _props_p, sentido_motor="infundado",
                           tipo_asunto="amparo_directo", huella_adelanto="H")
    ok(_det.get(_quien, {}).get("de") == "por_recalificar" and _det[_quien].get("se_estudia") == _motivo,
       f"{_rama}: el tumbado lleva el motivo por el que se estudia ({_motivo})")
    _rr = resultado(F3p)
    _pr_e, _acc_e = rc.entradas(_rr, _cr, _det)
    _a = next(a for a in _acc_e if a["problema"] == _quien)
    ok(_a["se_estudia"] == _motivo and "innecesario" not in rc.catalogo(True, False, _motivo)
       and "innecesario" in rc.catalogo(True, False, ""),
       f"{_rama}: la recalificación lo recibe y su catálogo no admite «innecesario»")
    _txt = rc.prompt(_rr, None, _pr_e, _acc_e)
    _tramo = _txt.split(f"pregunta: {_quien}")[1].split("PLANTEAMIENTO")[0]
    ok("por qué se estudia aunque el principal prospere:" in _tramo
       and "innecesario" not in _tramo.split("calificaciones admitidas:")[1].split("\n")[0],
       f"{_rama}: el prompt lo dice como dato y no le ofrece «innecesario»")
    _res, _faltas, _ = rc.validar({"planteamientos": [
        {"numero": _a["numero"], "sentido": "innecesario",
         "razon": "la premisa del secretario deja sin materia el estudio de este planteamiento"}]},
        [_a], _pr_e)
    ok(_quien not in _res and _faltas and "innecesario" in _faltas[0],
       f"{_rama}: la validación rechaza «innecesario» y lo vuelve a pedir")
    # DEFENSA en el árbol: una casilla guardada que lo trajera innecesario no se aplica.
    _k = rc.clave_de(_det)
    _cas = {"clave": _k, "resultados": {_quien: {"sentido": "innecesario", "razon": "sin materia por la premisa",
                                                 "presupone": None, "verificado": False}}}
    _cr2 = _crit_p()
    _, _det2 = ad.aplicar(copy.deepcopy(F3p), _cr2, [], _props_p, sentido_motor="infundado",
                          tipo_asunto="amparo_directo", huella_adelanto="H", recalificadas=_cas)
    _c2 = next(c for c in _cr2 if c["problema"] == _quien)
    ok(_det2[_quien]["de"] == "por_recalificar" and _c2["sentido"] != "innecesario",
       f"{_rama}: el árbol no aplica un «innecesario» que se colara")
ok(rc.VERSION != "recal-1", "la versión de la recalificación sube: lo guardado con el catálogo viejo no sirve")


print("\n8 · LA CLAVE DE LA RECALIFICACIÓN LLEVA LA SUPLENCIA CONFIRMADA Y EL CONTEXTO")
_SUP = {"fraccion": "V", "a_favor_de": "el trabajador", "confirmada": True}
_SUP_NO = dict(_SUP, confirmada=False)
_base_k = ("¿P?", "infundado", "r", ["¿A?"], "H", "amparo_directo")
ok(rc.huella_premisa(None, "") == "" and rc.huella_premisa(_SUP_NO, "") == ""
   and rc.huella_premisa(_SUP, "") != "" and rc.huella_premisa(None, "el contrato colectivo") != "",
   "la huella: vacía sin suplencia confirmada ni contexto; con cualquiera de las dos, no")
ok(rc.clave(*_base_k, rc.huella_premisa(_SUP, "")) != rc.clave(*_base_k, rc.huella_premisa(None, ""))
   and rc.clave(*_base_k, rc.huella_premisa(None, "otro contexto")) != rc.clave(*_base_k)
   and rc.clave_premisa("¿P?", "infundado", "r", "H", "amparo_directo", rc.huella_premisa(_SUP, ""))
   != rc.clave_premisa("¿P?", "infundado", "r", "H", "amparo_directo"),
   "clave y clave_premisa cambian al confirmar la suplencia o al aportar contexto")
# Por la puerta de verdad: lo recalificado antes de confirmar la suplencia no
# se reutiliza después (ni por la misma clave ni por la misma premisa).
_ses_s = {"propuestas": [types.SimpleNamespace(**p) for p in PRg], "material": None}
_arm_sin = NS["_taller_armar_criterio"](resultado(F3g), _ses_s, GLOBg, criterios_json=CJg)
_arm_con = NS["_taller_armar_criterio"](resultado(F3g), _ses_s, GLOBg, criterios_json=CJg, suplencia=_SUP)
_arm_ctx = NS["_taller_armar_criterio"](resultado(F3g), _ses_s, GLOBg, criterios_json=CJg,
                                        contexto="el contrato colectivo dice otra cosa")
_k_sin, _k_con = rc.clave_de(_arm_sin["detalle"]), rc.clave_de(_arm_con["detalle"])
ok(_k_sin and _k_con and _k_sin != _k_con and rc.clave_de(_arm_ctx["detalle"]) not in (_k_sin, _k_con),
   "_taller_armar_criterio: otra suplencia confirmada u otro contexto, otra clave")
_guard = {_k_sin: {"estado": "listo", "premisa": rc.premisa_de(_arm_sin["detalle"]), "hecho": 1.0,
                   "resultados": {P2g: {"sentido": "inoperante", "razon": "no combate la consideración toral",
                                        "presupone": None, "verificado": False},
                                  P3g: {"sentido": "inoperante", "razon": "no combate la consideración toral",
                                        "presupone": None, "verificado": False}}}}
ok(rc.casilla_de(_guard, _k_con, rc.premisa_de(_arm_con["detalle"]), rc.pendientes(_arm_con["detalle"])) is None
   and rc.casilla_de(_guard, _k_sin, rc.premisa_de(_arm_sin["detalle"]),
                     rc.pendientes(_arm_sin["detalle"])) is not None,
   "la «inoperante» hecha sin suplencia no se aplica con la suplencia confirmada")
_arm_con2 = NS["_taller_armar_criterio"](resultado(F3g), _ses_s, GLOBg, criterios_json=CJg, suplencia=_SUP,
                                         recalificadas=_guard)
ok(sorted(rc.pendientes(_arm_con2["detalle"])) == sorted([P2g, P3g]),
   "…y con ella el árbol los deja por recalificar")
_pr_k, _acc_k = rc.entradas(resultado(F3g), _arm_con["crit"], _arm_con["detalle"])
ok(rc.clave(_pr_k["problema"], _pr_k["sentido"], _pr_k["razon"], [a["problema"] for a in _acc_k],
            _te.huella_contraste(resultado(F3g)), "amparo_directo", rc.huella_premisa(_SUP, ""))
   == _k_con, "la clave del árbol es la que calcula la recalificación con esa suplencia")


print("\n9 · «ESTUDIAR JUNTOS»: LA v1 CON EL TEXTO DE LA v2, Y SIN ÓRDENES OPUESTAS")
_G1 = [f6.Criterio(problema="¿Es ilegal la valoración del dictamen pericial?", sentido="infundado",
                   razonamiento="r", jerarquia="principal", grupo="A"),
       f6.Criterio(problema="¿Se omitió valorar la testimonial de dos testigos?", sentido="infundado",
                   razonamiento="r", jerarquia="accesorio", grupo="A")]
_p_v1g = f6.prompt_estudio("acto", "conceptos", _G1, f6.Material(tipo_asunto="amparo_directo", variante="v1"))
_p_v2g = f6.prompt_estudio("acto", "conceptos", _G1, f6.Material(tipo_asunto="amparo_directo", variante="v2"))
_linea = lambda p: next(x for x in p.splitlines() if "SE ESTUDIA JUNTO" in x)  # noqa: E731
ok("no los contestes por separado" not in _p_v1g and "respuesta identificable" in _linea(_p_v1g)
   and _linea(_p_v1g) == _linea(_p_v2g),
   "(a) la v1 con grupo recibe el MISMO texto que la v2 (cada argumento, su respuesta)")
_G0 = [dataclasses.replace(c, grupo="") for c in _G1]
ok("SE ESTUDIA JUNTO" not in f6.prompt_estudio("acto", "conceptos", _G0,
                                               f6.Material(tipo_asunto="amparo_directo", variante="v1")),
   "(a) la v1 sin grupo no lo menciona (el resto lo congela test_prompt_v2)")
# (b) Después del árbol: el agrupado que queda sin materia sale del grupo.
Q1 = "¿La Sala valoró la testimonial ofrecida por la actora?"
Q2 = "¿Procedía la condena en costas de primera instancia?"
Q3 = "¿Se omitió estudiar la excepción de pago parcial?"
F3q = [{"pregunta": Q1, "jerarquia": "principal", "combate": "no valoró la testimonial"},
       {"pregunta": Q2, "jerarquia": "accesorio", "depende_de": 1, "combate": "las costas"},
       {"pregunta": Q3, "jerarquia": "accesorio", "combate": "la excepción de pago"}]
PRq = [_prop(Q1, "fundado"), _prop(Q2, "fundado"), _prop(Q3, "infundado")]


def _cjq(g3):
    return json.dumps([
        {"problema": Q1, "sentido": "fundado", "razonamiento": "no la valoró", "jerarquia": "principal",
         "tocado": True, "grupo": "A"},
        {"problema": Q2, "sentido": "fundado", "razonamiento": "r", "jerarquia": "accesorio", "tocado": False,
         "grupo": "A"},
        {"problema": Q3, "sentido": "infundado", "razonamiento": "r", "jerarquia": "accesorio", "tocado": False,
         "grupo": g3}], ensure_ascii=False)


_ses_q = {"propuestas": [types.SimpleNamespace(**p) for p in PRq], "material": None}
_arm_q = NS["_taller_armar_criterio"](resultado(F3q), _ses_q, {"sentido": "fundado", "alcanza": True},
                                      criterios_json=_cjq(""))
_g_q = {c.problema: (c.sentido, c.grupo) for c in _arm_q["crit"]}
ok(_g_q[Q2] == ("innecesario", "") and _g_q[Q1][1] == ""
   and any(a.startswith("EL GRUPO A NO SE APLICÓ") and Q2[:40] in a and "sin materia" in a
           for a in _arm_q["avisos_fases"]),
   "(b) el agrupado sin materia sale del grupo, el grupo de uno se deshace y se le dice")
_p_q = f6.prompt_estudio("acto", "conceptos", _arm_q["crit"],
                         f6.Material(tipo_asunto="amparo_directo", variante="v1"))
ok("SE ESTUDIA JUNTO" not in _p_q, "(b) el prompt ya no le da a la vez «se estudia junto» y «no se estudia»")
_arm_q3 = NS["_taller_armar_criterio"](resultado(F3q), _ses_q, {"sentido": "fundado", "alcanza": True},
                                       criterios_json=_cjq("A"))
_g_q3 = {c.problema: c.grupo for c in _arm_q3["crit"]}
ok(_g_q3 == {Q1: "A", Q2: "", Q3: "A"}
   and any(a.startswith("EL GRUPO A SE APLICÓ SIN") and Q2[:40] in a for a in _arm_q3["avisos_fases"]),
   "(b) con dos que quedan, el grupo sigue sin el que salió, y se le dice")
_arm_q0 = NS["_taller_armar_criterio"](resultado(F3q), _ses_q, {"sentido": "fundado", "alcanza": True},
                                       criterios_json=_cjq("").replace('"grupo": "A"', '"grupo": ""'))
ok(not any("EL GRUPO" in a for a in _arm_q0["avisos_fases"]), "(b) sin grupos, ningún aviso nuevo")


print("\n10 · LOS CONCEPTOS DE VIOLACIÓN VIENEN DEL FORMULARIO, TAMBIÉN VACÍOS")
# Los dos gemelos: el encargo en memoria traía los de una vuelta anterior.
_orig = (rc._llamar, _ra_t.resolver, _ra_t.resolver_en_vivo)
_ra_t.resolver, _ra_t.resolver_en_vivo = _resolver_espia, _en_vivo_espia
LLAMADAS["cv"] = []
try:
    rc._llamar = _llamar_bien
    for _gem in ("flujo", "plano"):
        _s, _r = _sesion()
        _r.encargo.conceptos_violacion = "LOS DE UNA VUELTA GLOBAL ANTERIOR"
        GN["_taller_recuperar_sesion"] = lambda u, n, _s=_s: _s
        if _gem == "flujo":
            asyncio.run(_flujo(GN))
        else:
            try:
                asyncio.run(GN["taller_resolver"](**_FORMg))
            except _Parar:
                pass
finally:
    rc._llamar, _ra_t.resolver, _ra_t.resolver_en_vivo = _orig
ok(LLAMADAS["cv"] == ["", ""], "los dos gemelos fijan los del formulario aunque lleguen vacíos")
# El plan: con formulario no cae al encargo; sin formulario (None), sí.
import plan_estudio as _pe_t  # noqa: E402
_ns_pe = {"_te": types.SimpleNamespace(huella_contraste=lambda r: "H")}
exec(compile(ast.Module(body=[FN["_taller_plan_entradas"]], type_ignores=[]), "main.py", "exec"), _ns_pe)
_orig_pe = (_pe_t.segmentos_de, _pe_t.huella_entradas)
try:
    _pe_t.segmentos_de = lambda *a, **k: [{"id": "C1.a"}]
    _pe_t.huella_entradas = lambda r, m, cv, segs: cv
    _r_pe = resultado(F3g)
    _r_pe.encargo.conceptos_violacion = "VIEJOS"
    _e_form = _ns_pe["_taller_plan_entradas"](_r_pe, {"material": object()}, [], conceptos_violacion="")
    _e_none = _ns_pe["_taller_plan_entradas"](_r_pe, {"material": object()}, [])
finally:
    _pe_t.segmentos_de, _pe_t.huella_entradas = _orig_pe
ok(_e_form["conceptos_violacion"] == "" and _e_none["conceptos_violacion"] == "VIEJOS",
   "_taller_plan_entradas: el formulario vacío manda; sólo sin formulario se lee el encargo")
# El precálculo no se lanza si la propuesta pide los conceptos.
_PEDIDOS = []


async def _pedido_espia(*a, **k):
    _PEDIDOS.append(k.get("conceptos_violacion"))
    return {"estado": "en_curso", "clave": "k"}


_ns_pp = {"_taller_armar_criterio": lambda *a, **k: {"crit": []}, "_taller_plan_pedido": _pedido_espia,
          "_con_autos": lambda r, c: c, "HTTPException": HTTPException, "err": str,
          "print": lambda *a, **k: None}
exec(compile(ast.Module(body=[FN["_taller_plan_desde_propuesta"]], type_ignores=[]), "main.py", "exec"),
     _ns_pp)
_resp_pp = {"global": {"alcanza": True, "sentido": "fundado", "razon": "r"}}
_r_pp = resultado(F3g)
_r_pp.encargo.conceptos_violacion = "VIEJOS"
asyncio.run(_ns_pp["_taller_plan_desde_propuesta"]("x@y", "1/2026", _r_pp, {}, dict(_resp_pp,
                                                                                 necesita_conceptos=True)))
ok(_PEDIDOS == [], "el plan no se adelanta cuando la propuesta pide los conceptos de violación")
asyncio.run(_ns_pp["_taller_plan_desde_propuesta"]("x@y", "1/2026", _r_pp, {}, _resp_pp))
ok(_PEDIDOS == [""], "sin esa necesidad se adelanta con los de la pantalla (ninguno), no con los del encargo")


print("\n11 · UN WORKER MUERTO: EL GEMELO RELANZA DENTRO DE SU VENTANA (relojes falsos)")


class _Res:
    def __init__(self, data):
        self.data = data


class _Base:
    def __init__(self, plan):
        self.filas = [{"email": "x@y.mx", "expediente": "1/2026", "plan": plan}]

    def table(self, _):
        return _Q(self)


class _Q:
    def __init__(self, b):
        self.b, self.modo, self.datos, self.f = b, "select", None, []

    def select(self, *_):
        return self

    def update(self, datos):
        self.modo, self.datos = "update", datos
        return self

    def eq(self, col, val):
        self.f.append(("eq", col, val))
        return self

    def is_(self, col, val):
        self.f.append(("is", col, val))
        return self

    def limit(self, _):
        return self

    def _pasa(self, fila):
        for op, col, val in self.f:
            if col == "plan->>rev":
                v = str((fila.get("plan") or {}).get("rev")) if isinstance(fila.get("plan"), dict) else None
            elif col == "plan->rev":
                v = (fila.get("plan") or {}).get("rev") if isinstance(fila.get("plan"), dict) else None
            else:
                v = fila.get(col)
            if (op == "eq" and v != val) or (op == "is" and v is not None):
                return False
        return True

    def execute(self):
        filas = [f for f in self.b.filas if self._pasa(f)]
        if self.modo == "update":
            for f in filas:
                f.update(copy.deepcopy(self.datos))
        return _Res([copy.deepcopy(f) for f in filas])


class _Reloj:
    """El tiempo sólo avanza cuando alguien duerme: la espera de 90 s dura nada."""
    def __init__(self, t):
        self.t = t

    def time(self):
        return self.t


def _asyncio_falso(reloj):
    class _A:
        def __getattr__(self, n):
            return getattr(asyncio, n)

        async def sleep(self, s):
            reloj.t += s
            await asyncio.sleep(0)

        async def to_thread(self, f, *a, **k):
            return f(*a, **k)
    return _A()


T0 = 1_000_000.0
# (a) La recalificación: la pantalla la lanzó y su worker murió 5 s antes de
# que el gemelo empezara a esperarla (la casilla quedó «en curso»).
_r_w = resultado(F3g)
_ses_w = {"propuestas": [types.SimpleNamespace(**p) for p in PRg], "material": None}
_arm_w = NS["_taller_armar_criterio"](_r_w, _ses_w, GLOBg, criterios_json=CJg)
_k_w, _h_w = rc.clave_de(_arm_w["detalle"]), _arm_w["huella"]
_doc_w, _ = rc.fila_pedir(None, _k_w, _h_w, T0 - 5)
_reloj = _Reloj(T0)
_ns_w = entorno(supabase_admin=_Base(_doc_w), time=_reloj, asyncio=_asyncio_falso(_reloj))
_orig_ll = rc._llamar
try:
    rc._llamar = _llamar_bien
    _out_w = asyncio.run(_ns_w["_taller_recalificar_para"]("x@y.mx", "1/2026", _r_w, _ses_w, GLOBg,
                                                           {"criterios_json": CJg}, _arm_w))
finally:
    rc._llamar = _orig_ll
ok(rc.ABANDONADA_S < rc.TOPE_S and rc.ABANDONADA_S >= 2 * rc.LATIDO_S,
   f"recalificación: abandono ({rc.ABANDONADA_S:.0f} s) < espera del gemelo ({rc.TOPE_S:.0f} s), "
   f"y más de dos latidos ({rc.LATIDO_S:.0f} s)")
ok(_out_w["estado"] == "listo" and not rc.pendientes(_out_w["arm"]["detalle"])
   and _reloj.t - T0 < rc.TOPE_S,
   f"recalificación: el gemelo la da por muerta y la relanza dentro de su ventana "
   f"(a los {_reloj.t - T0:.0f} s)")
# (b) El plan: lo mismo con la espera del resolver.
import plan_estudio as _pe_w  # noqa: E402
_reloj_p = _Reloj(T0)
_doc_p, _ = _pe_w.fila_pedir(None, "KP", "HP", T0 - 5)
_LANZADO = []


def _lanzar_espia(*a, **k):
    _LANZADO.append(_reloj_p.t - T0)
    f = asyncio.get_event_loop().create_future()
    f.set_result((None, ["el planificador no devolvió nada (doble)"]))
    return f


_ns_p = {"time": _reloj_p, "asyncio": _asyncio_falso(_reloj_p), "print": lambda *a, **k: None,
         "err": str, "supabase_admin": _Base(_doc_p), "_taller_plan_aplica": lambda r: True,
         "_taller_plan_entradas": lambda *a, **k: {"clave": "KP", "huella": "HP"},
         "_taller_plan_lanzar": _lanzar_espia}
exec(compile(ast.Module(body=[FN[n] for n in ("_taller_plan_cas", "_taller_plan_leer", "_taller_plan_para")],
                        type_ignores=[]), "main.py", "exec"), _ns_p)
_r_p = resultado(F3g)
asyncio.run(_ns_p["_taller_plan_para"]("x@y.mx", "1/2026", _r_p, {}, []))
ok(_pe_w.PLAN_ABANDONADO_S < _pe_w.ESPERA_RESOLVER_S and _pe_w.PLAN_ABANDONADO_S >= 2 * _pe_w.LATIDO_S,
   f"plan: abandono ({_pe_w.PLAN_ABANDONADO_S:.0f} s) < espera del resolver ({_pe_w.ESPERA_RESOLVER_S:.0f} s)")
ok(len(_LANZADO) == 1 and _LANZADO[0] < _pe_w.ESPERA_RESOLVER_S,
   "plan: el resolver lo da por muerto y lo relanza dentro de su ventana"
   + (f" (a los {_LANZADO[0]:.0f} s)" if _LANZADO else ""))
_src_pc = ast.get_source_segment(SRC_MAIN, FN["_taller_plan_correr"])
ok("timeout=_pe.LATIDO_S" in _src_pc, "la corrida del plan late al ritmo que supone su umbral")


print("\n12 · LA CITA DE «PRESUPONE» SE PIDE COMO UN PASAJE, NO COMO EL PLANTEAMIENTO")


class _M5i:
    tipo_asunto = "amparo_directo"
    sondeo = None
    tesis = []
    normas = []
    espejo = None


_p5i = " ".join(f5.prompt_propuesta(copy.deepcopy(F3g), _M5i(), "acto", "conceptos", False, "", "").split())
_i12 = _p5i[_p5i.find("12. LA SUERTE CONDICIONAL"):_p5i.find("13. LAS CONSTANCIAS")]
ok("PASAJE breve" in _i12 and "cuarenta palabras" in _i12 and "nunca todo lo que se combate" in _i12
   and "pasaje literal breve" in _p5i,
   "fase 5, instrucción 12 y esquema: un pasaje breve (hasta unas cuarenta palabras), nunca el entero")
_pr_i = {"problema": P1g, "sentido": "infundado", "razon": "r", "fase3": F3g[0]}
_acc_i = [{"problema": P2g, "numero": 2, "fase3": F3g[1], "procesal": False}]
_p_rc = " ".join(rc.prompt(resultado(F3g), None, _pr_i, _acc_i).split())
ok("PASAJE breve" in _p_rc and "cuarenta palabras" in _p_rc and "nunca todo lo que se combate" in _p_rc,
   "recalificación: la misma descripción")
_comb_i = ("la sala debió estudiar los agravios sobre la tasa de los intereses moratorios porque el pagaré "
           "no era título ejecutivo y la vía debía improceder desde la demanda")
_acc_e = [{"problema": P2g, "numero": 2, "fase3": dict(F3g[1], combate=_comb_i), "procesal": False}]
_, _, _av_e = rc.validar({"planteamientos": [{"numero": 2, "sentido": "inoperante",
                                              "razon": "descansa en la improcedencia de la vía que se desestimó",
                                              "presupone": {"cita": _comb_i, "por_que": "la vía improcedente"}}]},
                         _acc_e, _pr_i)
ok(_av_e.get(P2g) and "todo lo que se combate" in _av_e[P2g] and "no consta" not in _av_e[P2g],
   "el aviso de la cita entera no dice «no consta» (sí consta: es demasiado)")
print()
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN" if not FALLAS else f"FALLAN {len(FALLAS)}: " + " · ".join(FALLAS))
raise SystemExit(1 if FALLAS else 0)
