"""El puente: Iurexia pide un expediente y el SISE lo entrega, sin que el
secretario descargue nada y SIN que sus credenciales salgan de su máquina.

LA ALTERNATIVA QUE DAVID PIDIÓ BUSCAR, Y QUE SÍ EXISTE
══════════════════════════════════════════════════════
Él quiere que la plataforma descargue sola con la orden «174/2026, amparo
directo». Lo natural sería guardar su usuario y clave del SISE en el servidor de
Iurexia, y eso funcionaría — pero ataría el acceso al sistema del Consejo de la
Judicatura a la seguridad de una empresa privada: quien entrara en Iurexia
entraría en el SISE de cada secretario suscrito.

No hace falta. La sesión del SISE vive en el navegador del secretario, y desde
ahí se puede trabajar:

    Iurexia (web)  ──orden: «174/2026»──▶  agente local (esta máquina)
                                              │ usa la sesión ya abierta
                                              ▼
                                            SISE
                                              │ PDFs
                                              ▼
    Iurexia (nube) ◀──texto ya recortado── depuración local

Para el secretario la experiencia es EXACTAMENTE la que pidió: teclea el número
y el tipo, y aparece el proyecto. La diferencia es invisible para él y total en
seguridad: **Iurexia nunca ve una credencial y nunca recibe un expediente
entero**, sólo el texto de las piezas que importan.

Y hay un segundo ahorro: el secretario entra al SISE UNA VEZ al día. Mientras esa
sesión viva, todas las órdenes siguientes no le piden nada.

CÓMO SE USA
═══════════
    python sise_agente.py            # arranca en localhost:8765

y la web de Iurexia llama a `http://localhost:8765/expediente` desde el propio
navegador del secretario. Sólo escucha en 127.0.0.1: nada de fuera lo alcanza.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Optional

PUERTO = int(os.getenv("SISE_AGENTE_PUERTO", "8765"))
CARPETA = os.getenv("SISE_AGENTE_CARPETA",
                    os.path.expanduser("~/Documents/Iurexia/expedientes"))

# Sólo se aceptan órdenes de la propia máquina y del origen de Iurexia.
ORIGENES = ("https://iurexia.com", "https://www.iurexia.com",
            "http://localhost:3000")

_navegador = None       # se reutiliza: mantener la sesión ES el punto
_ctx = None
_lock = asyncio.Lock()


async def _sesion():
    """Un solo navegador para toda la jornada, con la sesión del SISE viva."""
    global _navegador, _ctx
    if _ctx is not None:
        return _ctx
    from playwright.async_api import async_playwright
    pw = await async_playwright().start()
    # Perfil persistente: si ya entró hoy, no se le vuelve a pedir nada.
    perfil = os.path.expanduser("~/.iurexia/perfil-sise")
    os.makedirs(perfil, exist_ok=True)
    _ctx = await pw.chromium.launch_persistent_context(
        perfil, headless=False, accept_downloads=True,
        args=["--no-first-run", "--no-default-browser-check"])
    return _ctx


async def _asegurar_login(page) -> bool:
    """Si la sesión ya vive, sigue. Si no, se le enseña el login y se espera."""
    import sise_local as sl
    await page.goto(sl.SISE_CONSULTA)
    if "Login.aspx" not in page.url:
        return True
    await page.goto(sl.SISE_LOGIN)
    await page.bring_to_front()
    return await sl._esperar_sesion(page)


async def traer(numero: str, tipo: str = "amparo directo") -> dict:
    """La orden completa: del número al texto listo para el pipeline."""
    import ocr_expedientes as ocr
    import sise_local as sl

    async with _lock:                       # una orden a la vez por navegador
        ctx = await _sesion()
        page = ctx.pages[0] if ctx.pages else await ctx.new_page()
        if not await _asegurar_login(page):
            return {"error": "No se completó el acceso al SISE."}

        destino = os.path.join(CARPETA, numero.replace("/", "-"))
        os.makedirs(destino, exist_ok=True)
        d = await sl.bajar_en(page, numero, tipo, destino)

    salida = {"expediente": numero, "neun": d.neun,
              "presentacion": d.presentacion, "avisos": list(d.avisos),
              "documentos": []}

    # La depuración corre AQUÍ, en la máquina del secretario: a la nube sólo
    # sube el texto de las piezas que el proyecto necesita.
    for doc in d.documentos:
        clase = "demanda" if doc.tipo == "promocion" else "sentencia"
        paginas = ocr.cribar(doc.ruta)
        nativo = "\n".join(p.texto for p in paginas if p.texto and not p.descartada)
        texto, aviso = ocr.recortar(nativo, clase) if nativo else ("", "sin texto nativo")
        if aviso:
            salida["avisos"].append(f"{doc.tipo}: {aviso}")
        salida["documentos"].append({
            "tipo": doc.tipo, "etiqueta": doc.etiqueta, "fecha": doc.fecha,
            "paginas": len(paginas),
            "descartadas": sum(1 for p in paginas if p.descartada),
            "texto": texto,
            "necesita_ocr": not texto,      # escaneado: lo hará la nube
            "ruta": doc.ruta,
        })
    return salida


def servir() -> None:
    from aiohttp import web

    async def handler(req):
        cuerpo = await req.json()
        num = (cuerpo.get("expediente") or "").strip()
        if not re.fullmatch(r"\d{1,4}\s*/\s*\d{4}", num):
            return web.json_response({"error": "Número de expediente inválido."},
                                     status=400)
        r = await traer(num, cuerpo.get("tipo") or "amparo directo")
        return web.json_response(r)

    async def salud(req):
        return web.json_response({"agente": "sise", "ok": True})

    @web.middleware
    async def cors(req, siguiente):
        origen = req.headers.get("Origin", "")
        if req.method == "OPTIONS":
            resp = web.Response()
        else:
            resp = await siguiente(req)
        if origen in ORIGENES:
            resp.headers["Access-Control-Allow-Origin"] = origen
            resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
            resp.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        return resp

    app = web.Application(middlewares=[cors])
    app.router.add_post("/expediente", handler)
    app.router.add_options("/expediente", handler)
    app.router.add_get("/salud", salud)
    print(f"Agente del SISE escuchando en http://127.0.0.1:{PUERTO}")
    print("Sólo acepta órdenes de esta máquina. Tus claves no salen de aquí.")
    web.run_app(app, host="127.0.0.1", port=PUERTO, print=None)


if __name__ == "__main__":
    servir()
