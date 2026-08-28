"""Descarga los documentos de un expediente del SISE. CORRE EN LA MAC, NO EN LA NUBE.

LA REGLA DE SEGURIDAD, QUE ES EL DISEÑO ENTERO
══════════════════════════════════════════════
Este módulo **NO recibe, NO guarda y NO transmite credenciales**. Abre un Chrome
visible en la pantalla del secretario, lo lleva al login del SISE y SE DETIENE.
Él teclea su usuario y su clave en la ventana del navegador, contra el servidor
del Consejo de la Judicatura, igual que cualquier día. El programa espera a que
la sesión exista y sigue desde ahí.

Comprometer Iurexia no puede dar acceso al SISE de nadie, porque Iurexia nunca
tiene nada que dar. Y a la nube sólo sube TEXTO YA RECORTADO, nunca el
expediente entero: menos exposición de datos de las partes y menos coste de OCR.

LO QUE SE APRENDIÓ MIRANDO EL SISE POR DENTRO (28-ago-2026, expediente 174/2026)
════════════════════════════════════════════════════════════════════════════════
· Es ASP.NET WebForms: no hay URLs de descarga, todo es `__doPostBack` sobre una
  sesión con ViewState. Obliga a navegador real; un cliente HTTP no sirve.
· El campo del número lleva MÁSCARA («00/0000») que revierte cualquier escritura
  sintética. Se esquiva con el setter nativo de HTMLInputElement.
· La rejilla del expediente tiene nombres de control sistemáticos:
      ctl00$MainContentPlaceHolder$grvPanelCentral$ctl{NN}${tipo}
  con tipo ∈ imgPromocion (escrito de la parte) · imgDetJud (determinación
  judicial) · imgAcuDetJud (acuse) · imgNotConsCon (notificación).
  `ctl02` es la primera fila.
· El visor abre `Controles/DescargaArchivos.aspx` mediante `modalWin()`.

PARA EL ADELANTO SÓLO HACEN FALTA TRES: la promoción de la primera fila —que en
amparo directo trae la demanda Y el acto reclamado en el mismo escaneo—, su
determinación —el auto de admisión, de donde salen las fechas— y la última
determinación, que es el turno. Lo demás —alegatos, acuses, notificaciones— es
justo la basura que no hay que pagarle a ningún OCR.
"""

from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass, field
from typing import Optional

SISE_LOGIN = "https://sise.cjf.gob.mx/sise/Login.aspx?ReturnUrl=%2fsise%2findex.aspx"
SISE_CONSULTA = "https://sise.cjf.gob.mx/Sise/ExpedienteElectronico/DefaultConsulta.aspx"

# Los del combo de la propia pantalla, leídos del SISE.
TIPOS = {
    "amparo directo": "10", "amparo en revision": "11",
    "conflicto competencial": "12", "impedimento": "13",
    "revision contenciosa": "14", "queja": "15", "revision fiscal": "16",
    "reclamacion": "20", "inconformidad": "25", "varios": "27",
}

ESPERA_LOGIN = 300          # cinco minutos para que teclee con calma


@dataclass
class Documento:
    fila: int
    tipo: str                 # promocion | determinacion
    etiqueta: str             # «Admisión», «Amparo directo,»
    fecha: str
    ruta: str = ""


@dataclass
class Descarga:
    expediente: str
    neun: str = ""
    documentos: list[Documento] = field(default_factory=list)
    avisos: list[str] = field(default_factory=list)


async def _esperar_sesion(page) -> bool:
    """Hasta que el secretario haya entrado. No se le mete prisa ni se toca."""
    for _ in range(ESPERA_LOGIN):
        try:
            if "Login.aspx" not in page.url:
                if await page.query_selector("#ctl00_CurrentLoginStatus_ctl00"):
                    return True
        except Exception:
            pass
        await asyncio.sleep(1)
    return False


async def _poner_expediente(page, numero: str) -> None:
    """El campo lleva máscara JS: se escribe con el setter nativo."""
    await page.evaluate(
        """(n) => {const i=document.getElementById(
             'ctl00_MainContentPlaceHolder_txtNoExpAsignado');
           const set=Object.getOwnPropertyDescriptor(
             window.HTMLInputElement.prototype,'value').set;
           set.call(i,n); i.dispatchEvent(new Event('change',{bubbles:true}));}""",
        numero)


async def _filas(page) -> list[dict]:
    """Lo que hay en la rejilla del expediente, con su índice de control."""
    return await page.evaluate("""() => {
        const t=[...document.querySelectorAll('table')]
                 .filter(x=>x.innerText.includes('Determinación Judicial')).pop();
        if(!t) return [];
        return [...t.querySelectorAll('tr')].map(tr=>{
            const td=[...tr.querySelectorAll('td')].map(x=>x.innerText.trim());
            const im=[...tr.querySelectorAll('input[type=image]')].map(x=>x.id);
            return td.length ? {celdas:td, controles:im} : null;
        }).filter(Boolean);}""")


def _elegir(filas: list[dict]) -> list[tuple[str, str, str, str]]:
    """(id_control, tipo, etiqueta, fecha) — sólo lo que alimenta el adelanto.

    Se descarta por NOMBRE lo que se sabe inútil, en vez de bajarlo todo y
    filtrar después: cada página que no se baja es una que no se paga.
    """
    fuera = []
    for i, f in enumerate(filas):
        fecha = f["celdas"][0] if f["celdas"] else ""
        etiqueta = " ".join(f["celdas"][2:5]) if len(f["celdas"]) > 2 else ""
        for cid in f["controles"]:
            if cid.endswith("imgPromocion") and i == 0:
                fuera.append((cid, "promocion", etiqueta, fecha))
            elif cid.endswith("imgDetJud") and (i == 0 or i == len(filas) - 1):
                fuera.append((cid, "determinacion", etiqueta, fecha))
    return fuera


async def bajar(numero: str, tipo_asunto: str = "amparo directo",
                destino: str = ".", visible: bool = True) -> Descarga:
    """El circuito completo. `numero` como «174/2026»."""
    from playwright.async_api import async_playwright

    d = Descarga(expediente=numero)
    os.makedirs(destino, exist_ok=True)

    async with async_playwright() as pw:
        # headless=False A PROPÓSITO: el secretario tiene que VER dónde teclea
        # su contraseña. Un login invisible es indistinguible de un robo.
        navegador = await pw.chromium.launch(headless=not visible)
        ctx = await navegador.new_context(accept_downloads=True)
        page = await ctx.new_page()
        try:
            await page.goto(SISE_LOGIN)
            print("→ Teclea tu usuario y clave en la ventana del navegador.")
            if not await _esperar_sesion(page):
                d.avisos.append("No se completó el acceso al SISE.")
                return d

            await page.goto(SISE_CONSULTA)
            await page.select_option("#ctl00_MainContentPlaceHolder_ddlTipoAsunto",
                                     TIPOS.get(tipo_asunto.lower(), "10"))
            await _poner_expediente(page, numero)
            await page.click("#ctl00_MainContentPlaceHolder_btnBuscarExpediente")
            await page.wait_for_load_state("networkidle")

            texto = await page.inner_text("body")
            m = re.search(r"Único Nacional:\s*([\d]+)", texto)
            if m:
                d.neun = m.group(1)

            # El cuaderno se abre haciendo clic en su carátula.
            car = await page.query_selector("input[type=image][src*='caratula']")
            if car:
                await car.click()
                await page.wait_for_load_state("networkidle")

            filas = await _filas(page)
            if not filas:
                d.avisos.append("El expediente no tiene rejilla de documentos.")
                return d

            for cid, tipo, etiqueta, fecha in _elegir(filas):
                try:
                    async with page.expect_download(timeout=45000) as espera:
                        await page.click("#" + cid)
                    dl = await espera.value
                    ruta = os.path.join(destino, f"{numero.replace('/','-')} "
                                                 f"{tipo} {fecha.replace('/','-')}.pdf")
                    await dl.save_as(ruta)
                    d.documentos.append(Documento(0, tipo, etiqueta, fecha, ruta))
                except Exception as e:
                    d.avisos.append(f"No se pudo bajar {tipo} de {fecha}: "
                                    f"{type(e).__name__}")
        finally:
            await ctx.close()
            await navegador.close()
    return d
