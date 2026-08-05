"""
Capa web multiagente, anclada a Google y SUBORDINADA a la ley indexada.

─── QUÉ CAMBIÓ Y POR QUÉ ───────────────────────────────────────────────────
La primera versión era UNA sola llamada con una instrucción que le pedía al
modelo callarse («responde SIN NOVEDADES si no encuentras nada reciente»).
Resultado medido en producción: la etapa web sólo aparecía en consultas de sed
noticiosa, y cuando aparecía devolvía ccn-law.com, fralla.mx, hklaw.com y
kpmg.com — blogs de despachos y consultoras. Justo lo que este módulo existe
para evitar.

Ahora son TRES AGENTES EN PARALELO, cada uno con una misión distinta y su
propio coto de dominios:

  vigencia   → ¿el ordenamiento tuvo reformas? DOF, Cámara, Senado.
  criterios  → ¿hay criterios o comunicados nuevos? SCJN, CJF, Semanario.
  local      → ¿qué dice el congreso o el poder judicial de la entidad?

Corren a la vez, así que los tres juntos tardan lo que el más lento.

─── LA REGLA QUE NO SE NEGOCIA: FILTRO DURO ────────────────────────────────
Antes los dominios oficiales sólo se ORDENABAN primero. Si Google no devolvía
ninguno, se colaban los cuatro blogs. Ahora lo no oficial se DESCARTA: si un
agente no encuentra fuente oficial, no aporta nada.

Es deliberado y es caro en cobertura, pero lo que distingue a Iurexia es citar
el artículo con su fuente verificable. Un abogado que vea un blog de despacho
citado junto a un artículo de ley deja de creerle a todo lo demás.

─── SIEMPRE SE INFORMA ─────────────────────────────────────────────────────
La etapa se emite corra como corra: si los tres agentes vuelven de vacío, se
dice «sin cambios recientes». Que el usuario vea que se consultó y no había
nada es información; que la etapa desaparezca sin explicación, no.
"""

import asyncio
import os
import re
from typing import Any, Dict, List, Optional

# Interruptor. Sin esto el módulo no hace nada: se apaga desde Render sin
# desplegar si sale caro o ruidoso.
WEB_ACTIVA = os.getenv("BUSQUEDA_WEB_ACTIVA", "false").lower() in ("1", "true", "si", "sí")
WEB_MODELO = os.getenv("BUSQUEDA_WEB_MODELO", "gemini-3-flash-preview")
# 22s por agente. MEDIDO en producción, no estimado: con 15s los tres agentes
# expiraban («[vigencia] timeout», «[criterios] timeout», «[local] timeout» en
# los registros) y la capa devolvía «sin novedades» en el 100% de las consultas.
# El anclaje a Google no es una llamada normal: busca, descarga las páginas y
# luego genera, y eso son 15-20s de forma habitual.
#
# Los tres corren en paralelo, así que el conjunto tarda lo que el más lento.
# La tarea arranca ANTES del RAG (~12s) y quien la consume le da 18s más de
# gracia: presupuesto total ~30s, holgado sobre los 22.
WEB_TIMEOUT = float(os.getenv("BUSQUEDA_WEB_TIMEOUT", "22"))

# Marca que viaja en el marcador cuando se consultó y no había nada nuevo.
SIN_NOVEDADES = "__sin_novedades__"

# ── Los cotos de cada agente ────────────────────────────────────────────────
# `gob.mx` cubre las dependencias federales; los congresos estatales viven en
# dominios propios y se añaden por sufijo.
OFICIALES_FEDERALES = (
    "dof.gob.mx", "diputados.gob.mx", "senado.gob.mx", "gob.mx",
    "ordenjuridico.gob.mx",
)
OFICIALES_JUDICIALES = (
    "scjn.gob.mx", "sjf.scjn.gob.mx", "sitios.scjn.gob.mx", "cjf.gob.mx",
    "te.gob.mx", "tfja.gob.mx",
)
# Cualquier dominio de gobierno o poder judicial estatal.
PATRON_ESTATAL = re.compile(r"\.(gob|poderjudicial)\.mx$|congreso[a-z]*\.gob\.mx$")

AGENTES = (
    {
        "id": "vigencia",
        "etiqueta": "Vigencia y reformas",
        "cotos": OFICIALES_FEDERALES,
        "mision": (
            "Localiza el ordenamiento aplicable en su fuente oficial e indica su "
            "ESTADO DE VIGENCIA: fecha de la última reforma publicada en el "
            "Diario Oficial de la Federación. Busca en dof.gob.mx y en la página "
            "de leyes federales de la Cámara de Diputados. Si hubo reformas "
            "recientes, dilo; si el texto lleva años sin cambios, dilo también."
        ),
    },
    {
        "id": "criterios",
        "etiqueta": "Criterios recientes",
        "cotos": OFICIALES_JUDICIALES,
        "mision": (
            "Localiza en los sitios del Poder Judicial de la Federación lo que "
            "haya publicado sobre el tema: criterios, jurisprudencia, tesis, "
            "comunicados o material de consulta. Busca en scjn.gob.mx, "
            "sjf.scjn.gob.mx y cjf.gob.mx. No hace falta que sea reciente: "
            "interesa lo más autorizado que exista."
        ),
    },
    {
        "id": "local",
        "etiqueta": "Ámbito local",
        "cotos": (),          # se resuelve por patrón estatal
        "mision": (
            "Localiza lo que el congreso del estado o su poder judicial hayan "
            "publicado sobre el tema: el ordenamiento local aplicable, reformas, "
            "el periódico oficial del estado o acuerdos del tribunal superior. "
            "No hace falta que sea reciente."
        ),
    },
)


def _dominio(url: str) -> str:
    m = re.match(r"https?://([^/]+)", url or "")
    return m.group(1).replace("www.", "") if m else ""


def _es_oficial(dominio: str, cotos: tuple) -> bool:
    """Oficial = está en el coto del agente, o es un dominio de gobierno."""
    d = (dominio or "").lower()
    if not d:
        return False
    if any(d == c or d.endswith("." + c) for c in cotos):
        return True
    return bool(PATRON_ESTATAL.search(d))


async def _un_agente(agente: dict, consulta: str, estado: Optional[str]) -> Dict[str, Any]:
    """Lanza un agente. Nunca lanza excepción: ante cualquier fallo, vacío."""
    vacio = {"id": agente["id"], "resumen": "", "fuentes": []}
    try:
        from main import get_gemini_client, get_gemini_model_name
        from google.genai import types as gtypes

        donde = f" en el estado de {estado}" if estado else ""
        if agente["id"] == "local" and not estado:
            return vacio      # sin entidad, este agente no tiene qué buscar

        instruccion = (
            f"Consulta jurídica mexicana: {consulta}{donde}\n\n"
            f"TU MISIÓN: {agente['mision']}\n\n"
            "Responde en dos o tres frases, en español, sin adornos y sin "
            "repetir la consulta.\n\n"
            "IMPORTANTE: no exijas que la información sea reciente. Casi ninguna "
            "consulta jurídica trata de una novedad, y pedir novedad hacía que "
            "esta capa devolviera NADA casi siempre. Basta con que la fuente sea "
            "OFICIAL y venga al caso. Responde NADA sólo si de verdad no "
            "encuentras ninguna fuente oficial pertinente."
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
        if not texto or texto.upper().startswith("NADA"):
            return vacio

        # Las URLs vienen en los metadatos de anclaje, no en el texto: así se
        # muestran las de verdad y no las que el modelo pudiera inventar.
        # Ojo: Gemini entrega una URL de redirección de vertexaisearch; el
        # dominio real viaja en el `title`.
        fuentes, vistos = [], set()
        for cand in (getattr(resp, "candidates", None) or []):
            meta = getattr(cand, "grounding_metadata", None)
            for trozo in (getattr(meta, "grounding_chunks", None) or []):
                web = getattr(trozo, "web", None)
                url = getattr(web, "uri", "") if web else ""
                if not url:
                    continue
                titulo = (getattr(web, "title", "") or "").strip()
                es_dom = bool(re.fullmatch(r"[a-z0-9.-]+\.[a-z]{2,}", titulo.lower()))
                dom = titulo.lower() if es_dom else _dominio(url)

                # FILTRO DURO: lo que no es oficial no entra. Antes sólo se
                # ordenaba, y cuando Google no devolvía nada oficial se colaban
                # blogs de despachos.
                if not _es_oficial(dom, agente["cotos"]) or dom in vistos:
                    continue
                vistos.add(dom)
                fuentes.append({"titulo": (titulo or dom)[:140], "url": url,
                                "dominio": dom, "agente": agente["id"]})

        if not fuentes:
            return vacio      # sin respaldo oficial, el resumen no vale nada
        return {"id": agente["id"], "resumen": texto[:600], "fuentes": fuentes[:3]}

    except asyncio.TimeoutError:
        print(f"   🌐 [{agente['id']}] timeout ({WEB_TIMEOUT}s)")
        return vacio
    except Exception as e:
        print(f"   🌐 [{agente['id']}] falló ({type(e).__name__}: {str(e)[:80]})")
        return vacio


async def buscar_en_web(consulta: str, estado: Optional[str] = None) -> Dict[str, Any]:
    """
    Lanza los tres agentes en paralelo y funde lo que traigan.

    Devuelve {"resumen", "fuentes", "agentes": [ids que aportaron], "corrio": bool}.
    Nunca lanza: ante cualquier fallo la consulta sigue su curso sin web.
    """
    vacio = {"resumen": "", "fuentes": [], "agentes": [], "corrio": False}
    if not WEB_ACTIVA or not consulta.strip():
        return vacio

    try:
        resultados = await asyncio.gather(
            *[_un_agente(a, consulta, estado) for a in AGENTES],
            return_exceptions=True,
        )
    except Exception as e:
        print(f"   🌐 La capa web falló entera ({type(e).__name__}) — se sigue sin ella")
        return vacio

    partes, fuentes, aportaron = [], [], []
    for r in resultados:
        if isinstance(r, Exception) or not isinstance(r, dict) or not r.get("resumen"):
            continue
        etiqueta = next(a["etiqueta"] for a in AGENTES if a["id"] == r["id"])
        partes.append(f"[{etiqueta}] {r['resumen']}")
        fuentes.extend(r["fuentes"])
        aportaron.append(r["id"])

    # `corrio` en True aunque no haya nada: la etapa se pinta igual y se dice
    # «sin cambios recientes». Que desaparezca sin explicación es peor.
    return {
        "resumen": "\n\n".join(partes),
        "fuentes": fuentes[:6],
        "agentes": aportaron,
        "corrio": True,
    }


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
        "Lo de abajo proviene de una búsqueda en fuentes oficiales en línea.",
        "NO es la ley y NO sustituye a los artículos del contexto documental.",
        "Úsalo sólo para advertir de reformas o publicaciones recientes. Si",
        "contradice un artículo del contexto documental, MANDA EL ARTÍCULO, y",
        "puedes señalar que hay información en línea que apunta a un cambio.",
        "NUNCA cites una fuente de aquí como fundamento de una afirmación",
        "jurídica: el fundamento son los artículos y las tesis verificadas.",
        "",
        web["resumen"],
        "",
        "Fuentes consultadas:",
    ]
    for f in web.get("fuentes", []):
        lineas.append(f"  · {f['dominio']} — {f['titulo']}")
    lineas.append("</contexto_web>")
    return "\n".join(lineas)
