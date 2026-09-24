"""La búsqueda web con OpenAI (gpt-6-luna), sin tocar la red — 23-sep-2026.

    .venv/bin/python test_web_openai.py
"""
import asyncio
import os

import busqueda_web as bw
import fase_internet as fi
import web_openai as wo

FALLOS = []


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


# La forma real de /v1/responses con web_search e include=sources.
RESPUESTA = {
    "status": "completed",
    "output": [
        {"type": "web_search_call", "action": {"sources": [
            {"url": "https://sjf2.scjn.gob.mx/detalle/tesis/2029549"},
            {"url": "https://precedentelegal.com/tesis/2016903"}]}},
        {"type": "message", "content": [{"type": "output_text",
         "text": ("- **2018 — registro digital: 2016903**; 2a./J. 46/2018 (10a.), **ACTOS, OPERACIONES "
                  "O SERVICIOS BANCARIOS. SU BLOQUEO ES CONSTITUCIONAL** "
                  "([scjn.gob.mx](https://www.scjn.gob.mx/files/AR-1214-2016-170901.pdf?utm_source=openai))\n"
                  "- registro digital: 2029549, 2a./J. 101/2024 (11a.)"),
         "annotations": [
             {"type": "url_citation", "url": "https://www.scjn.gob.mx/files/AR-1214-2016-170901.pdf?utm_source=openai",
              "title": "Amparo en revisión 1214/2016"},
             {"type": "url_citation", "url": "https://blog.ejemplo.com/bloqueo", "title": "Un blog"}]}]},
    ],
}

print("\n1 · LEER LA RESPUESTA")
r = wo.leer_respuesta(RESPUESTA)
ok("registro digital: 2016903" in r["texto"], "el texto del mensaje")
ok(len(r["citas"]) == 2 and r["citas"][0][1].startswith("https://www.scjn.gob.mx"), "las citas ancladas, con título")
ok(r["consultadas"] == ["https://sjf2.scjn.gob.mx/detalle/tesis/2029549", "https://precedentelegal.com/tesis/2016903"],
   "las páginas que abrió la búsqueda")
ok(not r["cortada"], "completa no es cortada")
ok(wo.leer_respuesta({"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}})["cortada"],
   "sin presupuesto de salida SÍ es cortada")

print("\n2 · LOS ENLACES FUERA DEL TEXTO")
limpio = wo.sin_enlaces(r["texto"])
ok("http" not in limpio and "utm_source" not in limpio, "sin URL ni rastreo")
ok("scjn.gob.mx" in limpio, "el texto del enlace se queda; se va la dirección")
ok("170901" not in fi._registros_del_texto(limpio), "el nombre de un PDF no es un registro (el 170901 medido)")
ok(fi._registros_del_texto(limpio) == ["2016903", "2029549"], "los registros de verdad sí salen")

print("\n3 · LA LÍNEA DE LA CORTE (fase_internet) CON EL MOTOR NUEVO")


async def _falso_buscar(instruccion, **kw):
    _falso_buscar.kw = kw
    return dict(wo.leer_respuesta(RESPUESTA), error=None, segundos=1.0, uso={})

_orig = wo.buscar
wo.buscar = _falso_buscar
try:
    a = asyncio.run(fi._un_angulo_openai("bloqueo de cuentas", "", "amparo en revisión", fi.ANGULOS[0]))
finally:
    wo.buscar = _orig
ok(a["registros"] == ["2016903", "2029549"], "registros limpios")
ok(any("2a./J. 46/2018" in c for c in a["claves"]), "y las claves")
ok(all(fi._es_de_la_corte(f["url"]) for f in a["fuentes"]) and a["fuentes"], "sólo fuentes de la Corte (el blog no)")
ok(_falso_buscar.kw.get("contexto") == "high" and _falso_buscar.kw.get("timeout") == fi.TIMEOUT_OPENAI_S,
   "con la configuración medida: página completa y su propio plazo")

print("\n4 · EL CHAT (busqueda_web) CON EL MOTOR NUEVO")
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY") or "sk-prueba"
os.environ["BUSQUEDA_WEB_MOTOR"] = "openai"
wo.buscar = _falso_buscar
try:
    c = asyncio.run(bw._consultar("¿qué dice la Corte?", 400, bw.OFICIALES_JUDICIALES))
finally:
    wo.buscar = _orig
ok("http" not in c["texto"], "el resumen llega sin enlaces")
ok(("", "https://sjf2.scjn.gob.mx/detalle/tesis/2029549") in c["crudas"], "lo consultado también cuenta como fuente")
ok("scjn.gob.mx" in (_falso_buscar.kw.get("dominios") or []), "el coto se pide desde la búsqueda")
ok(not bw._es_oficial("blog.ejemplo.com", bw.OFICIALES_JUDICIALES), "y el filtro duro sigue tirando el blog")

print("\n5 · EL INTERRUPTOR")
os.environ["BUSQUEDA_WEB_MOTOR"] = "openrouter"
ok(not wo.usar_openai(), "BUSQUEDA_WEB_MOTOR=openrouter vuelve a sonar sin desplegar")
ok(bw.espera_en_vivo() == 10.0, "y con sonar el chat espera lo de antes")
os.environ["BUSQUEDA_WEB_MOTOR"] = "openai"
ok(wo.usar_openai() and bw.espera_en_vivo() == 18.0, "con OpenAI, el motor y su espera")

print("\n" + ("TODO PASA" if not FALLOS else f"{len(FALLOS)} FALLAN: {FALLOS}"))
raise SystemExit(1 if FALLOS else 0)
