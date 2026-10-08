"""La memoria de la conversación — 7-oct-2026.

    .venv/bin/python test_memoria_por_plan.py

Sin red y sin gastar API. El caso: Aurora (pro, folio 1317-23) adjuntó su
contestación tres veces a mitad de una conversación y el análisis la leyó sin
nada de lo anterior —/analyze-document no recibía historial—; además todos los
planes tenían la misma memoria y nadie avisaba cuando se llenaba.

Se comprueba:
  1. el nivel de cada plan y su tope (Platinum = el techo del motor; cada plan
     menor, menos);
  2. el marcador MEMORIA_LLENA: forma, plan mayor, y que se limpia del
     historial como los demás marcadores de pantalla;
  3. que la conversación de Aurora (574,269 caracteres) cabe entera en
     Platinum y se recorta en Pro, Básico y Gratuito, sin pasarse del tope;
  4. el endpoint /analyze-document acepta `historial`;
  5. la regla de coherencia existe y se inyecta sólo con turnos previos;
  6. el agente de vigencia se abstiene con entidad y NO_APLICA, y sin
     entidad sigue igual.
"""
import asyncio
import inspect
import json
import os
import sys

sys.path.insert(0, os.getcwd())
import main  # noqa: E402
import busqueda_web as bw  # noqa: E402

fallos = []


def ok(cond, nombre):
    print(("  ✓ " if cond else "  ✗ ") + nombre)
    if not cond:
        fallos.append(nombre)


print("1. Niveles y topes")
ok(main.nivel_de_plan("pro_monthly") == "pro", "pro_monthly → pro")
ok(main.nivel_de_plan("pro_annual") == "pro", "pro_annual → pro")
ok(main.nivel_de_plan("basico_monthly") == "basico", "basico_monthly → basico")
ok(main.nivel_de_plan("platinum_annual") == "platinum", "platinum_annual → platinum")
ok(main.nivel_de_plan("ultra_secretarios") == "platinum", "ultra → platinum")
ok(main.nivel_de_plan("gratuito") == "gratuito", "gratuito → gratuito")
ok(main.nivel_de_plan(None) == "gratuito", "sin plan → gratuito")
ok(main.nivel_de_plan("gratuito", es_admin=True) == "platinum", "admin → platinum")
topes = [main.tope_de_historial(n) for n in ("gratuito", "basico", "pro", "platinum")]
ok(topes == sorted(topes) and len(set(topes)) == 4, f"cada plan mayor tiene más memoria {topes}")
ok(topes[-1] == main.HISTORIAL_MAX_CHARS, "Platinum tiene el techo del motor")

print("2. El marcador")
m = main.marcador_memoria_llena("pro", 400000, 574269)
datos = json.loads(m.split("MEMORIA_LLENA:", 1)[1].rsplit("-->", 1)[0])
ok(datos == {"plan": "pro", "tope": 400000, "usado": 574269, "mayor": "platinum"}, "pro → mayor platinum")
ok(json.loads(main.marcador_memoria_llena("platinum", 1, 2).split("MEMORIA_LLENA:", 1)[1]
              .rsplit("-->", 1)[0])["mayor"] is None, "platinum → sin plan mayor")
texto = "Respuesta.\n" + m + "\nSigue la respuesta."
limpio = main._limpiar_marcadores(texto)
ok("MEMORIA_LLENA" not in limpio and "Sigue la respuesta." in limpio, "se limpia del historial")

print("3. La conversación de Aurora")
# Los largos reales de sus 34 mensajes, sin marcadores (Supabase, 7-oct-2026).
largos = [1254, 17626, 323, 15304, 9035, 31882, 13391, 25404, 250, 20611, 239, 21237, 241, 18046,
          204, 20157, 64282, 35698, 240, 22744, 185, 14384, 63862, 15888, 267, 40741, 325, 17841,
          63714, 19919, 455, 13557, 512, 4451]
conv = [main.Message(role="user" if i % 2 == 0 else "assistant", content="x" * n)
        for i, n in enumerate(largos)]
total = sum(largos)
ok(total == 574269, f"total {total:,}")
for nivel in ("platinum", "pro", "basico", "gratuito"):
    tope = main.tope_de_historial(nivel)
    h, antes, despues = main._recortar_historial(conv, presupuesto=tope)
    if nivel == "platinum":
        ok(despues == antes, "platinum: cabe entera, sin aviso")
    else:
        ok(despues < antes and despues <= tope, f"{nivel}: recortada a {despues:,} ≤ {tope:,}")
    ok(len(h[-1].content) == largos[-1], f"{nivel}: el último turno va entero")

print("4. /analyze-document recibe la conversación")
firma = inspect.signature(main.analyze_document)
ok("historial" in firma.parameters, "parámetro historial")
fuente = inspect.getsource(main.analyze_document)
ok("*_historial_para(_modelo)" in fuente, "el historial entra en la llamada principal")
ok("*_historial_para(DOCUMENT_MODEL" in fuente, "y en la continuación por recitación")
ok("REGLA_COHERENCIA_HILO" in fuente, "con la regla de coherencia")

ok("historial_archivo" in firma.parameters, "parámetro historial_archivo (archivo)")
ok("await historial_archivo.read(" in fuente, "el archivo se lee antes de devolver el flujo")
# Por qué archivo y no texto: Starlette corta cada CAMPO DE TEXTO en 1 MiB.
from fastapi import FastAPI, File, Form, UploadFile  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
_app = FastAPI()


@_app.post("/x")
async def _x(historial: str = Form(None), historial_archivo: UploadFile = File(None)):
    datos = (await historial_archivo.read()) if historial_archivo else (historial or "").encode()
    return {"n": len(datos)}

_c = TestClient(_app)
_grande = json.dumps([{"role": "user", "content": "x" * 2_000_000}])
r_texto = _c.post("/x", data={"historial": _grande})
r_arch = _c.post("/x", files={"historial_archivo": ("historial.json", _grande.encode(), "application/json")})
ok(r_texto.status_code == 400, f"2 MB como texto: {r_texto.status_code} (lo corta Starlette)")
ok(r_arch.status_code == 200 and r_arch.json()["n"] == len(_grande), "2 MB como archivo: pasa entero")

print("5. La regla de coherencia")
ok("COHERENCIA CON ESTA CONVERSACIÓN" in main.REGLA_COHERENCIA_HILO, "existe")
for pieza in ("fijó la legislación", "ya aclaró", "contraparte", "Iurexia propuso antes", "insiste"):
    ok(pieza in main.REGLA_COHERENCIA_HILO, f"cubre: {pieza}")
chat = inspect.getsource(main)
ok("if len(request.messages) > 1:\n                    dynamic_injections.append(REGLA_COHERENCIA_HILO)" in chat,
   "/chat la inyecta sólo con turnos previos")
ok("presupuesto=_tope_mem" in chat and "yield marcador_memoria_llena(_nivel_mem" in chat,
   "/chat recorta con el tope del plan y avisa")

print("6. El agente de vigencia")
vigencia = next(a for a in bw.AGENTES if a["id"] == "vigencia")
vistos = {}


async def _falso(instruccion, max_tokens, dominios, respuesta="NO_APLICA"):
    vistos["instruccion"] = instruccion
    return {"texto": respuesta, "crudas": [("CNPCF", "https://www.diputados.gob.mx/LeyesBiblio/pdf/CNPCF.pdf")],
            "cortada": False, "error": None}

original = bw._consultar
try:
    bw._consultar = _falso
    r = asyncio.run(bw._un_agente(vigencia, "excepción de prescripción en juicio ordinario civil", "GUANAJUATO"))
    ok(r["fuentes"] == [] and r["resumen"] == "", "con entidad y NO_APLICA: no pinta fuentes")
    ok("ESTADO DE GUANAJUATO" in vistos["instruccion"] and "Código Nacional de Procedimientos Civiles"
       in vistos["instruccion"], "la instrucción lleva la entidad y el límite")

    async def _federal(i, t, d):
        return await _falso(i, t, d, respuesta="Código Nacional de Procedimientos Civiles y Familiares, DOF 2023.")
    bw._consultar = _federal
    r = asyncio.run(bw._un_agente(vigencia, "requisitos del CNPCF para la demanda", None))
    ok(r["fuentes"] and "LA CONSULTA ES DEL ESTADO" not in vistos["instruccion"], "sin entidad: igual que antes")
    local = next(a for a in bw.AGENTES if a["id"] == "local")
    bw._consultar = _falso
    asyncio.run(bw._un_agente(local, "prescripción", "GUANAJUATO"))
    ok("NO_APLICA" not in vistos["instruccion"], "el agente local no recibe el límite")
finally:
    bw._consultar = original

print()
if fallos:
    print(f"FALLAN {len(fallos)}: " + "; ".join(fallos))
    sys.exit(1)
print("TODO PASA")
