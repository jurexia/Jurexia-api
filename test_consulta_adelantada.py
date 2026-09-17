"""La consulta del acervo corre sola al terminar el adelanto — 17-sep-2026.

David: «entre más simplifiquemos el proceso del taller, mejor para el
secretario», y «adelante» a que «Buscar solución jurídica» deje de ser un paso
que espera. La consulta se lanza suelta al final de /taller/adelanto, se guarda
con su marca y su huella, y el botón la recoge si el secretario no aporta
contexto; con contexto, la repite con él.

    .venv/bin/python test_consulta_adelantada.py
"""
import asyncio

import taller_estado as te

FALLOS = []


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


def _espera(fn, docs, reloj=None, **kw):
    lecturas = list(docs)
    t = {"ahora": 1000.0}
    dormidas = []

    def leer():
        return lecturas.pop(0) if len(lecturas) > 1 else lecturas[0]

    async def dormir(seg):
        dormidas.append(seg)
        t["ahora"] += (reloj or seg)

    res = asyncio.run(fn("h1", leer, ahora=lambda: t["ahora"], dormir=dormir, **kw))
    return res, len(dormidas)


print("\n1 · LA ESPERA DE LA CONSULTA AUTOMÁTICA")
ok(_espera(te.esperar_consulta, [{"huella": "h1", "estado": "listo", "segundos": 31.2}]) == (True, 0),
   "lista y de este adelanto: el botón la usa sin esperar")
ok(_espera(te.esperar_consulta, [{"huella": "h1", "estado": "en_curso", "desde": 995.0},
                                 {"huella": "h1", "estado": "listo"}]) == (True, 1),
   "en curso: espera y la usa cuando llega")
ok(_espera(te.esperar_consulta, [{"huella": "OTRA", "estado": "listo"}]) == (False, 0),
   "de otro adelanto: se consulta aquí")
ok(_espera(te.esperar_consulta, [None]) == (False, 0), "sin marca (sesión anterior): se consulta aquí")
ok(_espera(te.esperar_consulta, [{"huella": "h1", "estado": "fallo"}]) == (False, 0), "falló: se consulta aquí")
ok(_espera(te.esperar_consulta, [{"huella": "h1", "estado": "en_curso", "desde": 1000.0 - 200}]) == (False, 0),
   "en curso desde hace más de tres minutos: el worker murió, no se espera")
res, n = _espera(te.esperar_consulta, [{"huella": "h1", "estado": "en_curso", "desde": 1000.0}], reloj=30.0)
ok(res is False and n == 3, f"nunca llega: se deja de esperar a los 90 s (durmió {n} veces)")
ok(_espera(te.esperar_contraste, [{"huella": "h1", "estado": "listo", "items": [{"p": 1}]}]) == ([{"p": 1}], 0),
   "y el contraste sigue funcionando sobre la misma espera")

print("\n2 · LAS PUERTAS")
m = open("main.py", encoding="utf-8").read()
i_gs = m.find("def _taller_guardar_sesion(")
ok(0 < i_gs < m.find('"huella": _te.huella_contraste(r),', i_gs) < m.find("def _taller_recuperar_sesion("),
   "el adelanto deja su huella en el estado")
i_ade = m.find('@app.post("/taller/adelanto")')
i_con = m.find('@app.post("/taller/consultar")')
ok(0 < i_ade < m.find("asyncio.ensure_future(_taller_preconsultar(user_email, numero, r))", i_ade) < i_con,
   "el adelanto lanza la consulta sola")
src_pre = m[m.find("async def _taller_preconsultar("):m.find("\n\n\n", m.find("async def _taller_preconsultar("))]
ok('"estado": "en_curso"' in src_pre and '"estado": "listo"' in src_pre and '"estado": "fallo"' in src_pre,
   "la consulta sola marca en_curso, listo y fallo")
ok("huella=huella" in src_pre and "avisos=list(r.avisos or [])" in src_pre,
   "y guarda material y avisos sólo si la fila sigue siendo de este adelanto")
ok('chat_client, "")' in src_pre, "sin contexto del secretario: es la consulta del botón, tal cual")
tramo = m[i_con:m.find('@app.post("/taller/contexto")', i_con)]
ok("if not _ctx:" in tramo and "await _te.esperar_consulta(" in tramo,
   "el botón usa la consulta sola SÓLO cuando no hay contexto")
ok(tramo.find("if material is None:") < tramo.find("material = await _ra.consultar(")
   and "chat_client, _ctx)" in tramo,
   "con contexto, o sin marca, consulta como siempre")
ok("avisos=list(r.avisos or [])" in tramo, "y ya persiste los avisos de la consulta manual")
src_gm = m[m.find("def _taller_guardar_material("):m.find("def _taller_guardar_marca(")]
ok('if huella and est.get("huella") != huella:' in src_gm and 'est["consulta"] = marca' in src_gm,
   "guardar el material respeta la huella y lleva la marca en la misma escritura")
ok("def _taller_guardar_marca(" in m and "def _taller_leer_marca(" in m
   and 'return _taller_guardar_marca(email, numero, "contraste", doc,' in m,
   "el contraste usa la misma marca genérica, con su huella")
ok('select(f"estado->{clave}")' in m, "las marcas se leen por su rama, no la fila entera")

print()
if FALLOS:
    print(f"FALLAN {len(FALLOS)}: " + " · ".join(FALLOS))
    raise SystemExit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
