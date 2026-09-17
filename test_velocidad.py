"""Las cuatro aceleraciones del 17-sep-2026, sin cambiar lo que se produce.

    .venv/bin/python test_velocidad.py
"""
import asyncio
import inspect
import time

import fase6_rag as rag
import fases123_pipeline as f123
import llamada_modelo as lm

FALLOS = []


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


print("\n1 · LA PRIMERA VUELTA RECIBE LAS TESIS A NOMBRAR")
escrito = ("Sirve de apoyo la jurisprudencia 2a./J. 139/2011 y la tesis 1a./J. 139/2005, "
           "así como la de registro 167062. " * 3)
p = f123.prompt_resumen_conceptos(escrito, True, "revision_fiscal")
ok("LAS TESIS QUE LA PARTE INVOCA" in p and "2A./J. 139/2011" in p and "167062" in p,
   "el prompt de agravios lista las tesis contadas sin modelo")
pa = f123.prompt_resumen_acto("La Sala se apoyó en la jurisprudencia 2a./J. 99/2007. " * 5, True, "revision_fiscal")
ok("LAS TESIS QUE LA AUTORIDAD INVOCA" in pa and "2A./J. 99/2007" in pa, "y el del acto, las de la autoridad")
ok(f123._bloque_tesis_a_nombrar(set(), "LA PARTE") == "", "sin tesis, no se añade nada")
ok("prompt_conceptos_a_fondo(" in inspect.getsource(f123.correr),
   "la reescritura a fondo sigue ahí como red")

print("\n2 · EL ACTO A FONDO CORRE EN PARALELO")
src = inspect.getsource(f123.correr)
ok("_tarea_acto = asyncio.ensure_future(_acto_a_fondo(ra))" in src, "se lanza como tarea")
ok(src.find("_tarea_acto = asyncio.ensure_future") < src.find("sin_resumir(texto_conceptos, rc"),
   "ANTES de la cobertura de los agravios, no después")
ok("await _tarea_acto" in src, "y se recoge")

print("\n3 · LA PETICIÓN DE RESPALDO")
lm.RESPALDO_MIN = lm.RESPALDO_MAX = 0.3


class _R:
    pass


class _Cli:
    def __init__(s, demoras):
        s.demoras, s.n = list(demoras), 0

    @property
    def chat(s):
        return s

    @property
    def completions(s):
        return s

    async def create(s, **kw):
        s.n += 1
        d = s.demoras.pop(0)
        await asyncio.sleep(d)
        r = _R()
        r.demora = d
        return r


async def _probar():
    c = _Cli([0.05])
    await lm.crear(c, model="m", messages=[], max_completion_tokens=100)
    ok(c.n == 1, "la llamada normal no lanza respaldo")
    c = _Cli([5, 0.1])
    t0 = time.perf_counter()
    r = await lm.crear(c, model="m", messages=[], max_completion_tokens=100)
    ok(c.n == 2 and r.demora == 0.1 and time.perf_counter() - t0 < 1.5,
       "la atascada se resuelve con el respaldo, sin esperar a la original")
    c = _Cli([0.5, 5])
    r = await lm.crear(c, model="m", messages=[], max_completion_tokens=100)
    ok(r.demora == 0.5, "si la original termina antes, se queda la original")

asyncio.run(_probar())
lm.RESPALDO_MIN, lm.RESPALDO_MAX = 180.0, 300.0
ok(lm.espera_normal({"max_completion_tokens": 3000}) == 180.0
   and lm.espera_normal({"max_completion_tokens": 12000}) == 300.0,
   "lo normal crece con el tamaño pedido, entre 180 y 300 s")

print("\n4 · LA CACHÉ DEL RESOLVEDOR")
src_r = inspect.getsource(rag.resolver_articulo)
ok("_CACHE_ARTICULOS" in src_r and "if not fallo:" in src_r,
   "se guarda la respuesta, pero no la de una consulta que falló")

print()
if FALLOS:
    print(f"FALLAN {len(FALLOS)}: " + " · ".join(FALLOS))
    raise SystemExit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
