"""BÚSQUEDA WEB CON OPENAI: el motor que sustituye a sonar.

═══ POR QUÉ (23-sep-2026) ═════════════════════════════════════════════════
Medido sobre doce problemas de siete materias, tres ángulos cada uno, con cada
registro cotejado contra la API oficial del Semanario y un jurado ciego de dos
proveedores (desempate y calibración con un tercero):

    motor                     precedentes pertinentes   USD/problema   espera
    gpt-6-luna + web_search          186 de 347             0,14         26 s
    gemini-3.8-flash + Google         93                    0,53         48 s
    Parallel / You.com          4 a 7 veces menos que OpenAI en los mismos pares
    perplexity/sonar            piloto: 1 registro donde gpt-5.6-luna trajo 5

Y la razón de fondo: sonar vive en OpenRouter, y David decidió irse de
OpenRouter poco a poco. Éste es el primer paso: la búsqueda web.

═══ LO QUE NO CAMBIA ══════════════════════════════════════════════════════
Quien llama sigue decidiendo qué se cita. Aquí sólo se busca y se devuelve el
texto con sus fuentes; el cotejo contra el acervo (fase_internet) y el filtro
duro de dominios oficiales (busqueda_web) siguen donde estaban.

Se vuelve a sonar sin desplegar con BUSQUEDA_WEB_MOTOR=openrouter.
"""
from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, List, Optional

MODELO = os.getenv("BUSQUEDA_WEB_OPENAI_MODELO", "gpt-6-luna")


def motor() -> str:
    return os.getenv("BUSQUEDA_WEB_MOTOR", "openai").strip().lower()


def usar_openai() -> bool:
    """OpenAI es el motor salvo que Render diga otra cosa o falte la clave."""
    return motor() == "openai" and bool(os.getenv("OPENAI_API_KEY", ""))


def hay_motor() -> bool:
    """¿Hay con qué buscar? La clave que hace falta depende del motor."""
    return usar_openai() or bool(os.getenv("OPENROUTER_API_KEY", ""))


# ═══ LOS ENLACES FUERA DEL TEXTO ═══════════════════════════════════════════
# OpenAI escribe las fuentes DENTRO de la respuesta, como enlaces Markdown con
# «?utm_source=openai». Dos daños medidos: el extractor de registros leyó
# «170901» del nombre de un PDF («AR-1214-2016-170901.pdf») como si fuera una
# tesis, y el filtro de texto normativo rechaza cualquier transcripción con
# Markdown. Las fuentes ya viajan aparte, en las anotaciones: del texto se van.
_RX_ENLACE_MD = re.compile(r"\[([^\]]*)\]\((https?://[^)\s]+)\)")
_RX_URL = re.compile(r"\(?\s*https?://[^\s)\]>\"']+\s*\)?")


def sin_enlaces(texto: str) -> str:
    t = _RX_ENLACE_MD.sub(lambda m: m.group(1), texto or "")
    t = _RX_URL.sub(" ", t)
    t = re.sub(r"\(\s*\)", "", t)
    return re.sub(r"[ \t]{2,}", " ", t).strip()


def leer_respuesta(d: Dict[str, Any]) -> Dict[str, Any]:
    """De la respuesta cruda de /v1/responses a lo que usan los módulos.

    citas       → (título, url) que el modelo ANCLÓ en su texto
    consultadas → todas las páginas que la búsqueda abrió (include=sources)
    """
    partes: List[str] = []
    citas: List[tuple] = []
    consultadas: List[str] = []
    for it in d.get("output") or []:
        if it.get("type") == "web_search_call":
            for s in ((it.get("action") or {}).get("sources") or []):
                if s.get("url"):
                    consultadas.append(s["url"])
        elif it.get("type") == "message":
            for ct in it.get("content") or []:
                if ct.get("type") != "output_text":
                    continue
                partes.append(ct.get("text") or "")
                for a in ct.get("annotations") or []:
                    if a.get("url"):
                        citas.append((a.get("title") or "", a["url"]))
    vistos, unicas = set(), []
    for t, u in citas:
        if u not in vistos:
            vistos.add(u)
            unicas.append((t, u))
    cortada = (d.get("status") == "incomplete"
               and (d.get("incomplete_details") or {}).get("reason") == "max_output_tokens")
    return {"texto": "\n".join(partes).strip(), "citas": unicas,
            "consultadas": list(dict.fromkeys(consultadas)), "cortada": cortada}


async def buscar(instruccion: str, *, contexto: str = "medium", esfuerzo: str = "low",
                 dominios: Optional[List[str]] = None, instrucciones: str = "",
                 max_salida: Optional[int] = None, timeout: float = 60.0) -> Dict[str, Any]:
    """Una búsqueda. Nunca lanza: el fallo vuelve en «error» y el texto vacío.

    contexto  → search_context_size: cuánto de cada página lee (low/medium/high)
    dominios  → filters.allowed_domains (incluye subdominios; hasta 100)
    """
    t0 = time.time()
    fuera = {"texto": "", "citas": [], "consultadas": [], "cortada": False,
             "error": None, "segundos": 0.0, "uso": {}}
    clave = os.getenv("OPENAI_API_KEY", "")
    if not clave:
        fuera["error"] = "falta OPENAI_API_KEY"
        return fuera
    herramienta: Dict[str, Any] = {"type": "web_search", "search_context_size": contexto,
                                   "user_location": {"type": "approximate", "country": "MX",
                                                     "timezone": "America/Mexico_City"}}
    if dominios:
        herramienta["filters"] = {"allowed_domains": [d.removeprefix("www.") for d in dominios][:100]}
    cuerpo: Dict[str, Any] = {"model": MODELO, "input": instruccion, "tools": [herramienta],
                              "include": ["web_search_call.action.sources"],
                              "reasoning": {"effort": esfuerzo}}
    if instrucciones:
        cuerpo["instructions"] = instrucciones
    if max_salida:
        cuerpo["max_output_tokens"] = max_salida
    try:
        import httpx
        async with httpx.AsyncClient(timeout=timeout) as cli:
            r = await cli.post("https://api.openai.com/v1/responses",
                               headers={"Authorization": f"Bearer {clave}"}, json=cuerpo)
        if r.status_code != 200:
            fuera["error"] = f"HTTP {r.status_code}: {r.text[:160]}"
        else:
            d = r.json()
            fuera.update(leer_respuesta(d))
            fuera["uso"] = d.get("usage") or {}
    except Exception as e:
        fuera["error"] = f"{type(e).__name__}: {str(e)[:120]}"
    fuera["segundos"] = round(time.time() - t0, 1)
    return fuera
