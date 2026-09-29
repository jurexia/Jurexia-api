# -*- coding: utf-8 -*-
"""Las escrituras del estado del taller, atómicas si está la migración y como antes si no.

    .venv/bin/python test_estado_atomico.py
"""
import ast, sys, types
FALLOS = []


def ok(c, q):
    print(f"   {'PASA ' if c else 'FALLA'}  {q}")
    if not c:
        FALLOS.append(q)


SRC = open("main.py", encoding="utf-8").read()
ARB = ast.parse(SRC)
FN = {n.name: n for n in ast.walk(ARB) if isinstance(n, ast.FunctionDef)}


class _Res:
    def __init__(self, data):
        self.data = data


class Falso:
    """Supabase de mentira: rpc con o sin la función, y la tabla de siempre."""
    def __init__(self, con_funcion=True, fila=None):
        self.con_funcion = con_funcion
        self.fila = fila if fila is not None else {"estado": {"huella": "H1", "consulta": {"x": 1}}}
        self.llamadas = []

    def rpc(self, nombre, args):
        self.llamadas.append(("rpc", nombre, args))
        falso = self

        class _Q:
            def execute(self_):
                if not falso.con_funcion:
                    raise Exception("PGRST202 Could not find the function public.taller_estado_parche")
                est = falso.fila["estado"]
                if args.get("p_huella") and est.get("huella") != args["p_huella"]:
                    return _Res(False)
                est.update(args["p_parche"])
                return _Res(True)
        return _Q()

    def table(self, nombre):
        falso = self

        class _T:
            def select(self_, *a): return self_
            def eq(self_, *a): return self_
            def limit(self_, *a): return self_
            def update(self_, d):
                falso.llamadas.append(("update", d))
                falso.fila["estado"] = d["estado"]
                return self_
            def execute(self_): return _Res([falso.fila])
        return _T()


def cargar(sb):
    ns = {"supabase_admin": sb, "err": lambda e: str(e), "print": lambda *a, **k: None,
          "_material_ligero": lambda m: {"tesis": [], "normas": []}}
    for f in ("_taller_parchar_estado", "_taller_guardar_marca", "_taller_guardar_material"):
        exec(compile(ast.Module(body=[FN[f]], type_ignores=[]), "main.py", "exec"), ns)
    return ns


print("\n1 · CON LA MIGRACIÓN: UNA SENTENCIA")
sb = Falso(True)
ns = cargar(sb)
ok(ns["_taller_guardar_marca"]("a@b.c", "1/2026", "propuesta", {"estado": "listo"}, "H1") is True
   and sb.fila["estado"]["propuesta"] == {"estado": "listo"} and sb.fila["estado"]["consulta"] == {"x": 1}
   and not any(c[0] == "update" for c in sb.llamadas),
   "la marca se mezcla sin reescribir el estado entero y no pisa las otras ramas")
ok(ns["_taller_guardar_marca"]("a@b.c", "1/2026", "propuesta", {"estado": "viejo"}, "OTRA") is False
   and sb.fila["estado"]["propuesta"] == {"estado": "listo"},
   "con otra huella no se escribe")
ok(ns["_taller_guardar_material"]("a@b.c", "1/2026", object(), "H1", marca={"estado": "listo"},
                                  avisos=["a"], otras={"decisiva": {"d": 1}}) is True
   and set(sb.fila["estado"]) >= {"material", "consulta", "avisos", "decisiva", "huella", "propuesta"},
   "el material, su marca, los avisos y las otras marcas, en la misma escritura")

print("\n2 · SIN LA MIGRACIÓN: COMO ANTES")
sb2 = Falso(False)
ns2 = cargar(sb2)
ok(ns2["_taller_guardar_marca"]("a@b.c", "1/2026", "propuesta", {"estado": "listo"}, "H1") is True
   and any(c[0] == "update" for c in sb2.llamadas),
   "si la función no existe, cae a la lectura-cambio-escritura de siempre (se puede desplegar antes que la migración)")

print("\n3 · LA MIGRACIÓN")
sql = open("migraciones/20260929_taller_estado_parche.sql", encoding="utf-8").read()
ok("security definer" in sql and "set search_path = public" in sql
   and "from anon" in sql and "from authenticated" in sql and "to service_role" in sql,
   "SECURITY DEFINER con search_path fijo, y EXECUTE sólo para service_role")
ok("estado->>'huella' = p_huella" in sql and "coalesce(estado, '{}'::jsonb) || p_parche" in sql,
   "la guarda de la huella y la mezcla van en el mismo UPDATE")

print()
if FALLOS:
    print("FALLAS:\n  " + "\n  ".join(FALLOS))
    sys.exit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
