"""El globo de internet no corría: leía _hilo_task antes de que existiera — 25-sep-2026.

    .venv/bin/python test_hilo_antes_que_la_web.py

Medido en Render ese día: las 7 consultas con el globo encendido desde el
despliegue de b1b3713 dejaron «🌐 No pude lanzar la búsqueda web: cannot access
local variable '_hilo_task' where it is not associated with a value», y ni un
agente web corrió. El bloque del globo pasaba `_hilo_task` a lanzar_agentes()
unas líneas ANTES de crearlo, y el `except` se lo tragaba. De paso se perdía la
tarea del acervo flojo, que vive en el mismo `try`.

Ni el compilador ni el portón lo ven: el nombre SÍ existe en el ámbito de
chat_endpoint (lo leen dos funciones anidadas), sólo que todavía no tiene
valor. Esto mira el ORDEN. Sin red: lee el árbol de main.py, no lo importa.
"""
import ast
import os
import sys

FALLOS = []


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


def propios(fn):
    """Los nodos del ámbito de `fn`, sin entrar en funciones ni clases de dentro."""
    fuera, pila = [], list(ast.iter_child_nodes(fn))
    while pila:
        x = pila.pop()
        fuera.append(x)
        if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        pila.extend(ast.iter_child_nodes(x))
    return fuera


def anidadas(fn):
    return [x for x in ast.walk(fn) if x is not fn
            and isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))]


def leidos_antes_de_nacer(fn):
    """Nombres que las anidadas leen de `fn` y que `fn` lee ANTES de asignarlos.

    Es la forma exacta del fallo: una variable de celda (la ve una anidada)
    leída por la propia función en una línea anterior a su primera asignación.
    Con ramas y bucles el orden de líneas no es el de ejecución, pero en el
    cuerpo de chat_endpoint, que corre de arriba abajo, sí lo es.
    """
    mios = propios(fn)
    asignados = {}
    for x in mios:
        if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store):
            asignados[x.id] = min(asignados.get(x.id, x.lineno), x.lineno)
    parametros = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
    de_celda = set()
    for h in anidadas(fn):
        for y in ast.walk(h):
            if isinstance(y, ast.Name) and y.id in asignados:
                de_celda.add(y.id)
    malos = []
    for x in mios:
        if (isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load)
                and x.id in de_celda and x.id not in parametros
                and x.lineno < asignados[x.id]):
            malos.append((x.id, x.lineno, asignados[x.id]))
    return sorted(set(malos), key=lambda m: m[1])


arbol = ast.parse(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py"),
                       encoding="utf8").read())
chat = next(n for n in ast.walk(arbol)
            if isinstance(n, ast.AsyncFunctionDef) and n.name == "chat_endpoint")

print("\n1 · _hilo_task EXISTE ANTES DE QUE LA WEB LO PIDA")
mios = propios(chat)
creado = min((x.lineno for x in mios if isinstance(x, ast.Name)
              and x.id == "_hilo_task" and isinstance(x.ctx, ast.Store)), default=None)
ok(creado is not None, "chat_endpoint crea _hilo_task en su propio ámbito")

lanzar = [x for x in mios if isinstance(x, ast.Call)
          and getattr(x.func, "id", getattr(x.func, "attr", None)) == "lanzar_agentes"]
ok(bool(lanzar), "el globo lanza los agentes con lanzar_agentes()")
for c in lanzar:
    usa_hilo = any(isinstance(a, ast.Name) and a.id == "_hilo_task" for a in c.args)
    if usa_hilo:
        ok(creado is not None and creado < c.lineno,
           f"lanzar_agentes(_hilo_task, …) en la línea {c.lineno} va después de crearlo "
           f"(línea {creado})")

retrieval = min((x.lineno for x in mios if isinstance(x, ast.Name)
                 and x.id == "retrieval_task" and isinstance(x.ctx, ast.Store)), default=None)
ok(creado is not None and retrieval is not None and creado < retrieval,
   "y antes de la búsqueda del acervo, que también lo espera")

print("\n2 · NINGUNA VARIABLE COMPARTIDA SE LEE ANTES DE NACER EN chat_endpoint")
malos = leidos_antes_de_nacer(chat)
for nombre, leida, nace in malos:
    print(f"      «{nombre}» se lee en la línea {leida} y nace en la {nace}")
ok(not malos, "todas las que ven las anidadas se asignan antes de leerse")

print("\n3 · LA COMPROBACIÓN SÍ VE EL FALLO (el código de antes del arreglo)")
roto = ast.parse(
    "async def chat_endpoint(request):\n"
    "    try:\n"
    "        tareas = lanzar_agentes(_hilo_task, request.estado)\n"
    "    except Exception as e:\n"
    "        print(e)\n"
    "    async def _perform_retrieval():\n"
    "        return await _hilo_task\n"
    "    _hilo_task = asyncio.create_task(algo())\n")
ok(leidos_antes_de_nacer(roto.body[0]) == [("_hilo_task", 3, 8)],
   "acusa _hilo_task leído en la 3 y creado en la 8")

print("\n" + ("TODO PASA" if not FALLOS else f"{len(FALLOS)} FALLAN: {FALLOS}"))
raise SystemExit(1 if FALLOS else 0)
