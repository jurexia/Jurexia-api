"""
Búsqueda web anclada a Google, como COMPLEMENTO de la ley indexada.

─── POR QUÉ ESTÁ SUBORDINADA ───────────────────────────────────────────────
Lo que distingue a Iurexia es que cita el artículo con su fuente oficial y
coteja las tesis contra el Semanario. Los primeros resultados de Google para
casi cualquier duda jurídica mexicana son blogs de despachos y notas
anteriores a la última reforma. Si un abogado ve que citamos un blog junto a
un artículo de ley, perdemos exactamente aquello por lo que nos paga.

De ahí las tres reglas que este módulo impone, y que no son negociables:

  1. La web NUNCA sustituye una cita ni cambia el texto de un artículo. Entra
     como contexto y así se le presenta al modelo.
  2. Se priorizan dominios oficiales. Lo que no lo sea entra marcado como
     secundario, y si la consulta no lo necesita, no entra.
  3. El bloque va rotulado aparte, para que nadie confunda una nota de prensa
     con el texto vigente.

Dónde sí aporta de verdad: reformas recientes, publicaciones del DOF de esta
semana, criterios apenas difundidos — todo aquello que el corpus indexado, por
definición, todavía no puede saber.
"""

import asyncio
import os
import re
from typing import Any, Dict, List, Optional

# Interruptor. Sin esto, el módulo no hace nada: permite apagarlo desde Render
# sin desplegar, si sale caro o ruidoso.
WEB_ACTIVA = os.getenv("BUSQUEDA_WEB_ACTIVA", "false").lower() in ("1", "true", "si", "sí")
WEB_MODELO = os.getenv("BUSQUEDA_WEB_MODELO", "gemini-3-flash-preview")
WEB_TIMEOUT = float(os.getenv("BUSQUEDA_WEB_TIMEOUT", "8"))

# Dominios que sí son fuente. El orden no importa; la pertenencia sí.
DOMINIOS_OFICIALES = (
    "dof.gob.mx", "scjn.gob.mx", "diputados.gob.mx", "senado.gob.mx",
    "cjf.gob.mx", "sitios.scjn.gob.mx", "sjf.scjn.gob.mx",
    "gob.mx", "congresooaxaca.gob.mx", "infomex.org.mx",
    "inai.org.mx", "tfja.gob.mx", "te.gob.mx", "ordenjuridico.gob.mx",
)


def _es_oficial(url: str) -> bool:
    u = (url or "").lower()
    return any(d in u for d in DOMINIOS_OFICIALES)


def _dominio(url: str) -> str:
    m = re.match(r"https?://([^/]+)", url or "")
    return m.group(1).replace("www.", "") if m else ""


async def buscar_en_web(consulta: str, estado: Optional[str] = None) -> Dict[str, Any]:
    """
    Devuelve {"resumen": str, "fuentes": [{"titulo","url","dominio","oficial"}]}.

    Nunca lanza: ante cualquier fallo devuelve vacío y la consulta sigue su
    curso sin web. Es un complemento, no una dependencia.
    """
    vacio = {"resumen": "", "fuentes": []}
    if not WEB_ACTIVA or not consulta.strip():
        return vacio

    try:
        from main import get_gemini_client, get_gemini_model_name
        from google.genai import types as gtypes

        contexto_estado = f" en el estado de {estado}" if estado else ""
        instruccion = (
            f"Busca en fuentes oficiales mexicanas información ACTUAL sobre: {consulta}{contexto_estado}.\n\n"
            "Prioriza el Diario Oficial de la Federación, la SCJN, el Congreso de la Unión, "
            "los congresos estatales y los poderes judiciales. "
            "Interesan sobre todo REFORMAS RECIENTES, publicaciones nuevas y criterios de los últimos meses.\n\n"
            "Responde en tres o cuatro frases, en español, sin adornos. "
            "Si no encuentras nada reciente y relevante, responde exactamente: SIN NOVEDADES."
        )

        cliente = get_gemini_client()

        def _llamar():
            return cliente.models.generate_content(
                model=get_gemini_model_name(WEB_MODELO),
                contents=instruccion,
                config=gtypes.GenerateContentConfig(
                    tools=[gtypes.Tool(google_search=gtypes.GoogleSearch())],
                    temperature=0.1,
                ),
            )

        resp = await asyncio.wait_for(asyncio.to_thread(_llamar), timeout=WEB_TIMEOUT)

        texto = (getattr(resp, "text", "") or "").strip()
        if not texto or "SIN NOVEDADES" in texto.upper():
            return vacio

        # Las URLs vienen en los metadatos de anclaje, no en el texto: así se
        # muestran las de verdad y no las que el modelo pudiera inventar.
        fuentes: List[Dict[str, Any]] = []
        vistos = set()
        for cand in (getattr(resp, "candidates", None) or []):
            meta = getattr(cand, "grounding_metadata", None)
            for trozo in (getattr(meta, "grounding_chunks", None) or []):
                web = getattr(trozo, "web", None)
                url = getattr(web, "uri", "") if web else ""
                if not url or url in vistos:
                    continue
                vistos.add(url)
                fuentes.append({
                    "titulo": (getattr(web, "title", "") or _dominio(url))[:140],
                    "url": url,
                    "dominio": _dominio(url),
                    "oficial": _es_oficial(url),
                })

        # Lo oficial primero; lo demás sólo para rellenar y siempre marcado.
        fuentes.sort(key=lambda f: (not f["oficial"], f["dominio"]))
        return {"resumen": texto[:1200], "fuentes": fuentes[:6]}

    except asyncio.TimeoutError:
        print(f"   🌐 Búsqueda web: timeout ({WEB_TIMEOUT}s) — se sigue sin ella")
        return vacio
    except Exception as e:
        print(f"   🌐 Búsqueda web falló ({type(e).__name__}: {str(e)[:90]}) — se sigue sin ella")
        return vacio


def bloque_para_prompt(web: Dict[str, Any]) -> str:
    """
    Convierte el resultado en un bloque para el prompt, con su jerarquía
    escrita de forma explícita: el modelo tiene que saber que esto NO es ley.
    """
    if not web or not web.get("resumen"):
        return ""

    lineas = [
        "<contexto_web>",
        "AVISO DE JERARQUÍA — LEE ESTO ANTES DE USAR LO QUE SIGUE:",
        "Lo de abajo proviene de una búsqueda en internet. NO es la ley y NO",
        "sustituye a los artículos del contexto documental. Úsalo sólo para",
        "advertir de reformas o publicaciones recientes. Si contradice un",
        "artículo del contexto documental, MANDA EL ARTÍCULO, y puedes señalar",
        "que existe información en línea que apunta a un cambio reciente.",
        "NUNCA cites una fuente de aquí como fundamento de una afirmación",
        "jurídica; para eso están la ley y las tesis verificadas.",
        "",
        web["resumen"],
    ]
    oficiales = [f for f in web.get("fuentes", []) if f["oficial"]]
    otras = [f for f in web.get("fuentes", []) if not f["oficial"]]
    if oficiales:
        lineas.append("\nFuentes oficiales consultadas:")
        lineas += [f"- {f['titulo']} ({f['dominio']})" for f in oficiales]
    if otras:
        lineas.append("\nFuentes NO oficiales (referencia, nunca fundamento):")
        lineas += [f"- {f['titulo']} ({f['dominio']})" for f in otras]
    lineas.append("</contexto_web>")
    return "\n".join(lineas)
