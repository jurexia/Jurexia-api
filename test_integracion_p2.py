# -*- coding: utf-8 -*-
"""Lo que la integración del Paso 2 añadió encima de las piezas (26-sep-2026).

Cada comprobación nombra el fallo que cierra:
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
import dataclasses
import types

import arbol_decision as ad
import fase5_propuesta as f5

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

print()
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN" if not FALLAS else f"FALLAN {len(FALLAS)}: " + " · ".join(FALLAS))
raise SystemExit(1 if FALLAS else 0)
