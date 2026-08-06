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

# ── EL MOTOR: perplexity/sonar vía OpenRouter ────────────────────────────
# Medido el 6-ago-2026 con la misma consulta de patria potestad:
#
#   perplexity/sonar (OpenRouter)     4.9s · 9 fuentes · 4 OFICIALES de Qro
#   perplexity/sonar-pro              5.1s · idéntico (y más caro)
#   gemini-3-flash-preview + anclaje  15-22s · busca cuando quiere (1/3 en
#                                     producción por contención de API)
#   gemini/gpt-5-mini con :online     0 fuentes (el plugin Exa no conoce
#                                     el derecho mexicano)
#
# Sonar SIEMPRE busca —es su única función— y devuelve las citas en la
# respuesta. Eso elimina de raíz la veleidad de Gemini que obligó a exigir
# URLs, reintentar y escalonar. A 5s por agente, además, caben las fuentes
# EN VIVO en el flujo.
WEB_MODELO = os.getenv("BUSQUEDA_WEB_MODELO", "perplexity/sonar")
WEB_TIMEOUT = float(os.getenv("BUSQUEDA_WEB_TIMEOUT", "14"))
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

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
# Organismos del Estado que NO están en .gob.mx y que el patrón se comía:
# la CNDH aparecía en los resultados y se descartaba como si fuera un blog.
OFICIALES_AUTONOMOS = (
    "cndh.org.mx", "inai.org.mx", "ine.mx", "senado.gob.mx",
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
    if any(d == c or d.endswith("." + c) for c in OFICIALES_AUTONOMOS):
        return True
    return bool(PATRON_ESTATAL.search(d))


async def _un_agente(agente: dict, consulta: str, estado: Optional[str]) -> Dict[str, Any]:
    """Un agente = una llamada a perplexity/sonar con su misión. Nunca lanza."""
    vacio = {"id": agente["id"], "resumen": "", "fuentes": []}
    try:
        import httpx

        donde = f" en el estado de {estado}" if estado else ""
        if agente["id"] == "local" and not estado:
            return vacio      # sin entidad, este agente no tiene qué buscar
        if not OPENROUTER_API_KEY:
            print("   🌐 Falta OPENROUTER_API_KEY — capa web sin motor")
            return vacio

        instruccion = (
            f"Consulta jurídica mexicana: {consulta}{donde}\n\n"
            f"TU MISIÓN: {agente['mision']}\n\n"
            "Responde en dos o tres frases, en español, sin adornos y sin "
            "repetir la consulta. Prioriza sitios oficiales mexicanos "
            "(.gob.mx, poderes judiciales, congresos)."
        )

        async with httpx.AsyncClient(timeout=WEB_TIMEOUT) as cli:
            r = await cli.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
                json={"model": WEB_MODELO,
                      "messages": [{"role": "user", "content": instruccion}],
                      "max_tokens": 400},
            )
            r.raise_for_status()
            d = r.json()

        msg = (d.get("choices") or [{}])[0].get("message", {}) or {}
        texto = (msg.get("content") or "").strip()

        # Las citas de sonar viajan en dos sitios según la versión de la API:
        # message.annotations (url_citation, con título) y el campo raíz
        # `citations` (lista de URLs). Se leen ambos.
        crudas = []
        for a in (msg.get("annotations") or []):
            uc = a.get("url_citation") or {}
            if uc.get("url"):
                crudas.append((uc.get("title") or "", uc["url"]))
        for u in (d.get("citations") or []):
            crudas.append(("", u))

        fuentes, vistos = [], set()
        for titulo, url in crudas:
            dom = _dominio(url)
            if not dom or dom in vistos:
                continue
            vistos.add(dom)
            # FILTRO DURO: lo que no es oficial no entra. leyes-mx.com y
            # justia.com aparecen SIEMPRE en estos resultados y jamás deben
            # pintarse junto a un artículo de ley.
            if not _es_oficial(dom, agente["cotos"]):
                continue
            fuentes.append({"titulo": (titulo or dom)[:140], "url": url,
                            "dominio": dom, "agente": agente["id"]})

        if not texto or not fuentes:
            print(f"   🌐 [{agente['id']}] {len(texto)} car., "
                  f"{len(crudas)} citas, {len(fuentes)} oficiales")
            return vacio
        return {"id": agente["id"], "resumen": texto[:600], "fuentes": fuentes[:3]}

    except Exception as e:
        print(f"   🌐 [{agente['id']}] falló ({type(e).__name__}: {str(e)[:80]})")
        return vacio


def lanzar_agentes(consulta: str, estado: Optional[str] = None) -> List[asyncio.Task]:
    """
    Lanza los agentes y devuelve sus TAREAS, sin esperarlas.

    Es la pieza que permite las fuentes EN VIVO: quien consume va recogiendo
    cada tarea conforme termina (FIRST_COMPLETED) y emite el marcador
    actualizado al frontend, en vez de esperar a que acabe la última.
    """
    if not WEB_ACTIVA or not consulta.strip():
        return []
    return [asyncio.create_task(_un_agente(a, consulta, estado)) for a in AGENTES]


async def buscar_en_web(consulta: str, estado: Optional[str] = None) -> Dict[str, Any]:
    """Envoltorio clásico: lanza los agentes y espera a todos. El camino
    nuevo (fuentes en vivo) usa lanzar_agentes() + fusionar() directamente."""
    tareas = lanzar_agentes(consulta, estado)
    if not tareas:
        return {"resumen": "", "fuentes": [], "agentes": [], "corrio": False}
    hechas, pendientes = await asyncio.wait(tareas, timeout=WEB_TIMEOUT + 2)
    for x in pendientes:
        x.cancel()
    resultados = []
    for x in hechas:
        try:
            resultados.append(x.result())
        except Exception:
            pass
    return fusionar(resultados)


def fusionar(resultados: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Funde lo que trajeron los agentes en un solo resultado."""
    partes, fuentes, aportaron = [], [], []
    for r in resultados:
        if not isinstance(r, dict) or not r.get("resumen"):
            continue
        etiqueta = next((a["etiqueta"] for a in AGENTES if a["id"] == r["id"]), r["id"])
        partes.append(f"[{etiqueta}] {r['resumen']}")
        fuentes.extend(r["fuentes"])
        aportaron.append(r["id"])
    return {"resumen": "\n\n".join(partes), "fuentes": fuentes[:6],
            "agentes": aportaron, "corrio": True}


def bloque_para_prompt(web: Dict[str, Any]) -> str:
    """
    Convierte el resultado en un bloque para el prompt, con su jerarquía
    escrita de forma explícita: el modelo tiene que saber que esto NO es ley.
    """
    if not web or not web.get("resumen"):
        return ""

    lineas = [
        "<contexto_web>",
        "INSTRUCCIONES PARA USAR LO QUE SIGUE (búsqueda en internet, dominios",
        "oficiales). El usuario PIDIÓ estas fuentes con un clic: espera verlas",
        "reflejadas en tu respuesta.",
        "",
        "1. INTEGRA lo relevante en el cuerpo de la respuesta, señalándolo:",
        "   «según información en línea de <dominio>, …». Que se distinga qué",
        "   viene de internet y qué del acervo documental.",
        "2. JERARQUÍA: esto NO es la ley y NO sustituye a los artículos del",
        "   contexto documental. Si contradice un artículo, MANDA EL ARTÍCULO,",
        "   y puedes señalar que hay información en línea que apunta a un",
        "   cambio reciente. El fundamento jurídico son los artículos y las",
        "   tesis verificadas, nunca una página web.",
        "3. NO añadas tu propia lista de fuentes web al final: el sistema",
        "   agrega la sección «Fuentes de internet consultadas» con los",
        "   enlaces exactos.",
        "",
        web["resumen"],
        "",
        "Fuentes consultadas:",
    ]
    for f in web.get("fuentes", []):
        lineas.append(f"  · {f['dominio']} — {f['titulo']}")
    lineas.append("</contexto_web>")
    return "\n".join(lineas)


def _tipo_de_sitio(dominio: str) -> str:
    """Naturaleza del organismo, a partir del dominio.

    Los portales oficiales mexicanos NO sirven /favicon.ico (se verificaron
    ocho: todos 404 o 403) y pedirle el icono a un servicio externo le
    contaría a un tercero qué está investigando cada abogado. Un icono por
    tipo de organismo es local, siempre pinta, y dice más que un favicon
    borroso de 16 píxeles.
    """
    d = (dominio or "").lower()
    if any(k in d for k in ("poderjudicial", "scjn", "cjf", "tribunal", "pjf", "sise", "juzgado")):
        return "judicial"
    if any(k in d for k in ("legislatura", "congreso", "diputados", "senado", "camara", "cámara")):
        return "legislativo"
    if any(k in d for k in ("dof", "periodicooficial", "sombradearteaga", "gaceta", "ordenjuridico", "normas")):
        return "oficial"
    if any(k in d for k in ("municipio", "ayuntamiento", "municipal")):
        return "municipal"
    if any(k in d for k in ("gob.mx", "fiscalia", "fiscalía", "segob", "cndh")):
        return "ejecutivo"
    return "web"


def bloque_fuentes_html(fuentes: List[Dict[str, Any]], nota: str, maximo: int = 6) -> str:
    """Las fuentes consultadas, como tarjetas HTML en UNA sola línea.

    Se emite HTML y no markdown por una razón medida: el renderizador de
    Iurexia (formatMarkdown en ChatMessage.tsx) nunca tuvo regla para
    `[texto](url)`, así que los enlaces salían crudos y además estirados por
    el `text-align: justify` de .prose-legal. Sin newlines dentro, porque el
    formateador convierte cada salto en <br/> y sólo respeta las líneas que
    empiezan por «<».
    """
    def esc(valor: Any) -> str:
        return (str(valor or "")
                .replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;").replace('"', "&quot;"))

    partes = ['<div class="fuentes-web">'
              '<div class="fw-cab"><span class="fw-globo">\U0001F310</span>'
              '<span>Fuentes de internet consultadas</span></div>'
              '<div class="fw-lista">']
    for f in (fuentes or [])[:maximo]:
        dominio = str(f.get("dominio", "") or "")
        url = str(f.get("url", "") or "")
        if not url:
            continue
        titulo = str(f.get("titulo") or dominio)
        titulo = " ".join(titulo.split())[:110] or dominio
        # NO se etiqueta el agente que la trajo: el buscador de «criterios»
        # devuelve a menudo dominios locales, y rotular «sanjoaquin.gob.mx ·
        # Criterios del PJF» es decirle al abogado algo falso.
        partes.append(
            f'<a class="fw-item" href="{esc(url)}" target="_blank" rel="noopener noreferrer">'
            f'<span class="fw-ico fw-ico--{_tipo_de_sitio(dominio)}"></span>'
            f'<span class="fw-txt"><span class="fw-tit">{esc(titulo)}</span>'
            f'<span class="fw-dom">{esc(dominio)}</span></span>'
            f'<span class="fw-flecha">&#8599;</span></a>'
        )
    partes.append(f'</div><div class="fw-nota">{esc(nota)}</div></div>')
    return "".join(partes)
