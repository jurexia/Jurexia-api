# -*- coding: utf-8 -*-
"""La consulta provisional: se propone igual, pero no se recomienda (rediseño, etapa 2).

    .venv/bin/python test_consulta_provisional.py
"""
import asyncio, json, sys, types
sys.path.insert(0, ".")
FALLOS = []


def ok(c, q):
    print(f"   {'PASA ' if c else 'FALLA'}  {q}")
    if not c:
        FALLOS.append(q)


import tarjeta_decision as td
import taller_estado as te
import fase6_estudio as f6
import contexto_taller as ct
# Cuenta de casa: la bandera «consulta_provisional» encendida. La propuesta por
# probabilidad (2-oct-2026) también nace «casa» y cambia esto: aquí se apaga
# para medir lo de siempre, y abajo se prueba encendida.
_SIN_PROB = {"banderas": {"propuesta_por_probabilidad": False}}
ct.poner(True, _SIN_PROB)

print("\n1 · LA TARJETA")
m = f6.Material()
m.consulta_estado = {"estado": "provisional", "faltan": ["decisiva"]}
x = td._provisional(m, "propuesta", "claro", ["Razón 1."])
ok(x["recomendada"] is None and x["estado"] == "no_alcanza" and x["estado_por_que"][0].startswith("Consulta PROVISIONAL")
   and x["estado_por_que"][1:] == ["Razón 1."],
   "provisional: no rotula recomendada, dice por qué y conserva las razones")
m.consulta_estado = {"estado": "completa", "faltan": []}
ok(td._provisional(m, "propuesta", "claro", ["R"]) == {"recomendada": "propuesta", "estado": "claro", "estado_por_que": ["R"]},
   "completa (o sin dato, sesión vieja): como siempre")
ok(td._provisional(f6.Material(), "opuesta", "reñido", []) ["recomendada"] == "opuesta", "sin el campo, como siempre")
ct.poner(False, {})
m.consulta_estado = {"estado": "provisional", "faltan": ["decisiva"]}
ok(td._provisional(m, "propuesta", "claro", ["R"])["recomendada"] == "propuesta",
   "para los de fuera, sin la bandera, la tarjeta como hoy (se mide antes)")
# CON LA PROPUESTA POR PROBABILIDAD (2-oct-2026, contrato E): la consulta
# provisional se dice, pero ya no quita la recomendación ni pone «no_alcanza».
ct.poner(True, {})
xp = td._provisional(m, "propuesta", "reñido", ["R"])
ok(xp["recomendada"] == "propuesta" and xp["estado"] == "reñido"
   and xp["estado_por_que"][0].startswith("Consulta PROVISIONAL") and xp["estado_por_que"][1:] == ["R"],
   "con la propuesta por probabilidad: se dice que es provisional y se sigue recomendando")
ct.poner(True, _SIN_PROB)

print("\n2 · VIAJA CON LA SESIÓN")
m2 = f6.Material()
m2.tesis = [{"registro": "1", "rubro": "R"}]
m2.consulta_estado = {"estado": "provisional", "faltan": ["decisiva"]}
m3 = te.material_rehidratado(json.loads(json.dumps(te.material_ligero(m2))))
ok(m3 is not None and m3.consulta_estado == {"estado": "provisional", "faltan": ["decisiva"]},
   "el estado de la consulta sobrevive a la base (otro worker lo ve)")

print("\n3 · EN MAIN")
src = open("main.py", encoding="utf-8").read()
ok(src.count("_taller_estado_consulta(") == 4, "se marca en las tres esperas de la decisiva (más su definición)")
ok('material.consulta_estado = {"estado": "completa", "faltan": []}' in src,
   "al sumar la figura, la consulta deja de ser provisional")
ok("consulta de {numero} completada al proponer" in src, "al proponer, si la decisiva ya llegó, se completa antes")


async def _marca():
    ns = {}
    import ast
    arbol = ast.parse(src)
    fn = next(n for n in ast.walk(arbol) if isinstance(n, ast.FunctionDef) and n.name == "_taller_estado_consulta")
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "main.py", "exec"), ns)
    mm = f6.Material()
    lenta = asyncio.ensure_future(asyncio.sleep(5))
    ns["_taller_estado_consulta"](mm, lenta, None)
    a = dict(mm.consulta_estado)
    lenta.cancel()
    hecha = asyncio.ensure_future(asyncio.sleep(0)); await hecha
    ns["_taller_estado_consulta"](mm, hecha, None)
    return a, dict(mm.consulta_estado)

a, b = asyncio.run(_marca())
ok(a["estado"] == "provisional" and b["estado"] == "completa",
   "provisional sólo si la decisiva sigue corriendo; si terminó (aunque sin formular), completa")

print()
if FALLOS:
    print("FALLAS:\n  " + "\n  ".join(FALLOS))
    sys.exit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
