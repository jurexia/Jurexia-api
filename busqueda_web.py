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

# ── DESDE EL 23-SEP-2026 EL MOTOR ES gpt-6-luna CON LA BÚSQUEDA DE OPENAI ──
# El paso uno de dejar OpenRouter (decisión de David). Lo medido y el porqué
# están en web_openai.py. Sonar sigue aquí, entero, para volver sin desplegar
# con BUSQUEDA_WEB_MOTOR=openrouter.
#
# gpt-6-luna tarda más que sonar. Cada agente pide poco —contexto «low», dos o
# tres frases— y su plazo es propio: los 14 s de sonar lo cortarían.
WEB_TIMEOUT_OPENAI = float(os.getenv("BUSQUEDA_WEB_OPENAI_TIMEOUT_CHAT", "25"))


def espera_en_vivo() -> float:
    """Cuánto espera el chat a los agentes DESPUÉS del RAG (arrancaron antes).
    Con sonar bastaban 10 s. Con gpt-6-luna se da más, porque la capa sólo
    corre cuando el abogado encendió el globo: pidió fuentes de internet."""
    import web_openai
    return float(os.getenv("BUSQUEDA_WEB_ESPERA_CHAT", "18" if web_openai.usar_openai() else "10"))


async def _consultar(instruccion: str, max_tokens: int, dominios: tuple = (),
                     contexto: str = "low") -> Dict[str, Any]:
    """Una pregunta al motor activo. Devuelve {texto, crudas, cortada, error}.

    Las «crudas» son (título, url) con lo que el motor dice haber consultado;
    el filtro duro de dominios oficiales se aplica DESPUÉS, igual para los dos
    motores."""
    import web_openai
    if web_openai.usar_openai():
        r = await web_openai.buscar(
            instruccion + "\n\nResponde en texto plano: sin Markdown y sin enlaces "
            "(las fuentes ya viajan aparte).",
            contexto=contexto, esfuerzo="low", dominios=list(dominios) or None,
            max_salida=max_tokens + 2500,       # el razonamiento cuenta dentro
            timeout=WEB_TIMEOUT_OPENAI)
        return {"texto": web_openai.sin_enlaces(r["texto"]),
                "crudas": r["citas"] + [("", u) for u in r["consultadas"]],
                "cortada": r["cortada"], "error": r["error"]}
    if not OPENROUTER_API_KEY:
        return {"texto": "", "crudas": [], "cortada": False, "error": "falta OPENROUTER_API_KEY"}
    import httpx
    async with httpx.AsyncClient(timeout=WEB_TIMEOUT) as cli:
        r = await cli.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
            json={"model": WEB_MODELO,
                  "messages": [{"role": "user", "content": instruccion}],
                  "max_tokens": max_tokens},
        )
        r.raise_for_status()
        d = r.json()
    msg = (d.get("choices") or [{}])[0].get("message", {}) or {}
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
    return {"texto": (msg.get("content") or "").strip(), "crudas": crudas,
            "cortada": ((d.get("choices") or [{}])[0].get("finish_reason") or "") == "length",
            "error": None}

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


# ── EL ACERVO QUE FALTA, BUSCADO EN INTERNET (24-sep-2026) ─────────────────
# David: «no en todos los estados tenemos buen acervo de ley. En esos estados
# donde hay pocos códigos habilita la búsqueda web a la par de luna para que
# otorgue más resultados. Sólo precisa que la fuente en esos casos es de
# internet».
#
# El agente «local» de arriba contesta en dos o tres frases: sirve para decir
# «el congreso publicó una reforma», no para suplir un código que no está.
# Éste pide lo que el acervo no tiene —el TEXTO de los artículos estatales
# aplicables, con su ley y su número— y sólo en sitios oficiales de la
# entidad. Lo que traiga viaja marcado como internet hasta la respuesta.
AGENTE_ACERVO_LOCAL = {
    "id": "acervo_local",
    "etiqueta": "Legislación estatal en internet",
    "cotos": (),          # se resuelve por patrón estatal, como «local»
    "max_tokens": 900,
    "tope_resumen": 2600,
    "tope_fuentes": 4,
    "mision": (
        "El acervo no tiene completa la legislación de esta entidad. Busca en los "
        "sitios OFICIALES del estado —congreso, periódico oficial, poder judicial— "
        "el texto VIGENTE de los artículos de la ley estatal que regulan lo que se "
        "consulta. Para cada artículo pertinente da el nombre exacto de la ley, el "
        "número de artículo y su texto literal o un extracto fiel. Hasta cinco "
        "artículos. Si no encuentras el texto oficial, dilo en una línea."
    ),
}


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


async def _un_agente(agente: dict, consulta, estado: Optional[str]) -> Dict[str, Any]:
    """Un agente = una llamada al buscador con su misión. Nunca lanza.

    `consulta` puede llegar como texto o como tarea que lo entrega: el chat la
    pasa así para que la búsqueda web espere la pregunta CON SU HILO (ver
    _consulta_con_hilo en main.py) sin retrasar el arranque de los agentes."""
    vacio = {"id": agente["id"], "resumen": "", "fuentes": []}
    try:
        if not isinstance(consulta, str):
            consulta = await consulta
        consulta = (consulta or "").strip()
        if len(consulta) > 400:
            consulta = consulta[:200] + " … " + consulta[-200:]
        if not consulta:
            return vacio
        donde = f" en el estado de {estado}" if estado else ""
        if agente["id"] in ("local", "acervo_local") and not estado:
            return vacio      # sin entidad, este agente no tiene qué buscar

        cierre = (
            "Responde en español, sin adornos y sin repetir la consulta, así: "
            "«Nombre de la ley — Artículo N: texto». Sólo sitios oficiales del estado."
            if agente["id"] == "acervo_local" else
            "Responde en dos o tres frases, en español, sin adornos y sin "
            "repetir la consulta. Prioriza sitios oficiales mexicanos "
            "(.gob.mx, poderes judiciales, congresos)."
        )
        instruccion = (
            f"Consulta jurídica mexicana: {consulta}{donde}\n\n"
            f"TU MISIÓN: {agente['mision']}\n\n{cierre}"
        )

        # Con OpenAI el coto se pide DESDE la búsqueda (allowed_domains): así
        # no se gastan las fuentes en blogs que el filtro duro tiraría. El
        # agente local no tiene coto fijo —su patrón es estatal— y busca
        # abierto; el filtro de abajo decide igual para todos.
        _dominios = (tuple(agente["cotos"]) + OFICIALES_AUTONOMOS) if agente["cotos"] else ()
        c = await _consultar(instruccion, agente.get("max_tokens", 400), _dominios)
        if c["error"]:
            print(f"   🌐 [{agente['id']}] sin motor o sin respuesta ({c['error'][:90]})")
            return vacio
        texto, crudas = c["texto"], c["crudas"]

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
        return {"id": agente["id"], "resumen": texto[:agente.get("tope_resumen", 600)],
                "fuentes": fuentes[:agente.get("tope_fuentes", 3)]}

    except Exception as e:
        print(f"   🌐 [{agente['id']}] falló ({type(e).__name__}: {str(e)[:80]})")
        return vacio


# ═══════════════════════════════════════════════════════════════════════════
# EL TEXTO DE UN ARTÍCULO QUE EL ACERVO NO TIENE
# ═══════════════════════════════════════════════════════════════════════════
# David, 16-sep-2026, sobre la revisión fiscal 2/2026: «salió "el motor no se
# atrevió con un sentido para todo el asunto" porque tuvo duda en dos por
# falta de acervo. Ésta es una oportunidad única para que un motor económico
# de búsqueda en internet traiga el artículo faltante (y así lo diga). Con
# esto el motor siempre propondrá la solución sí o sí».
#
# Tenía razón y el caso lo enseña entero: el problema principal de ese asunto
# era si el artículo 150, fracción XIX, del Reglamento Interior del IMSS
# faculta a una autoridad, y ese reglamento NO ESTÁ en ninguna colección. Sin
# el precepto, el modelo hizo lo que su prompt le manda —no inventar— y se
# abstuvo. Con el precepto delante puede proponer, y el sentido lo decide
# quien firma.
#
# TRES CANDADOS, porque esto entra en un proyecto de sentencia:
#   1. Sólo dominios OFICIALES: el mismo filtro duro de los demás agentes.
#      diputados.gob.mx y ordenjuridico.gob.mx publican los textos vigentes;
#      leyes-mx.com y justia.com no entran jamás.
#   2. Viaja MARCADO como `de_internet`, con su dominio y su URL, y el
#      compositor lo dice en la nota al pie. Nunca se confunde con el acervo.
#   3. Si no sale texto o no sale fuente oficial, no se devuelve nada: el
#      hueco declarado sigue siendo mejor que un texto de procedencia dudosa.
# LO QUE NO ES UN ARTÍCULO AUNQUE VENGA DE UN DOMINIO OFICIAL. Medido: con
# la «Ley Inventada de la Nada Absoluta» el buscador contestó «No puedo citar
# textualmente el artículo 77 de una ley inexistente…» y citó
# gaceta.diputados.gob.mx — dominio oficial, texto largo, sin dos puntos al
# final: pasaba los tres filtros y se habría colado como PRECEPTO en un
# proyecto de sentencia. Una negativa del buscador no es una norma.
_NEGATIVAS = re.compile(
    r"no\s+(?:puedo|pude|es\s+posible|encontr|localic|hay|existe|se\s+encontr|"
    r"dispongo|tengo\s+acceso)|"
    r"\bno\s+corresponde\b|\bno\s+identificable\b|\binexistente\b|"
    r"\bno\s+figura\b|\bsin\s+resultados\b|\blo\s+siento\b", re.I)

AGENTE_ARTICULO = {
    "id": "articulo",
    "etiqueta": "Texto del precepto",
    "cotos": OFICIALES_FEDERALES + OFICIALES_JUDICIALES,
}


async def texto_de_articulo(cuerpo_legal: str, numero: str,
                            estado: Optional[str] = None) -> Dict[str, Any]:
    """El texto literal de un artículo, traído de su fuente oficial en línea.

    Devuelve {"texto", "url", "dominio", "titulo"} o {} si no se pudo. Nunca
    lanza: quien llama sigue su camino con el hueco declarado, que es como
    estaba antes de esto.

    DOS FORMULACIONES, Y NO POR ADORNO. Medido contra el artículo 150 del
    Reglamento Interior del IMSS —el que dejó sin proponer a la revisión
    fiscal 2/2026—: con la misma pregunta, sonar contestó una vez «NO
    LOCALIZADO» y otra con el texto correcto citando imss.gob.mx,
    diputados.gob.mx, dof.gob.mx y ordenjuridico.gob.mx. La abstención no era
    del buscador: era de la formulación. La segunda vuelta pregunta como
    preguntaría una persona, que es la que funcionó.
    """
    vacio: Dict[str, Any] = {}
    ley = " ".join((cuerpo_legal or "").split())
    num = str(numero or "").strip()
    import web_openai
    if not WEB_ACTIVA or not ley or not num:
        return vacio
    if not web_openai.hay_motor():
        print("   🌐 Sin motor de búsqueda — no se busca el precepto")
        return vacio

    donde = f" (ámbito: {estado})" if estado else ""
    intentos = (
        (f"Transcribe el TEXTO VIGENTE del artículo {num} de esta norma "
         f"mexicana: {ley}{donde}. Cópialo TAL CUAL está publicado, COMPLETO "
         f"y CON TODAS SUS FRACCIONES si las tiene. Nada de resúmenes. "
         f"Búscalo en la fuente oficial: diputados.gob.mx, "
         f"ordenjuridico.gob.mx, dof.gob.mx o el sitio del organismo que "
         f"expide la norma. Si el artículo está derogado o reformado, dilo al "
         f"final. Si NO lo encuentras en una fuente oficial, responde "
         f"exactamente: NO LOCALIZADO."),
        (f"¿Qué dice el artículo {num} del {ley}{donde}? Cítalo textualmente y "
         f"completo, con sus fracciones, e indica la fuente oficial de donde "
         f"lo tomas."),
    )

    for i, instruccion in enumerate(intentos, 1):
        try:
            # El precepto se lee entero de la página oficial: contexto medio.
            c = await _consultar(instruccion, 1200,
                                 AGENTE_ARTICULO["cotos"] + OFICIALES_AUTONOMOS, contexto="medium")
            if c["error"]:
                print(f"   🌐 artículo {num} de «{ley[:40]}» · vuelta {i}: {c['error'][:90]}")
                continue
            texto = c["texto"]
            # UN ARTÍCULO CORTADO A LA MITAD DICE OTRA COSA. Si el modelo se
            # quedó sin presupuesto, lo que llega es media norma presentada
            # como norma entera.
            if c["cortada"]:
                print(f"   🌐 artículo {num} de «{ley[:40]}» · vuelta {i}: "
                      f"la respuesta se cortó por longitud")
                continue

            # UN ENCABEZADO NO ES UN ARTÍCULO. La primera vuelta devolvió
            # «Artículo 150.- Son atribuciones de las subdelegaciones, dentro
            # de su circunscripción territorial:» y ahí se acababa: el anuncio
            # de una lista de fracciones que no venía. Un precepto cortado en
            # los dos puntos, dentro de una sentencia, dice lo contrario de lo
            # que dice la norma.
            _plano = " ".join(texto.split())
            _corto = len(_plano) < 120
            _colgado = texto.rstrip().endswith((":", "…", "..."))
            # Y TIENE QUE SER UNA TRANSCRIPCIÓN, no una respuesta sobre ella:
            # el artículo se nombra en la cabecera de lo que se transcribe.
            _nombra = bool(re.search(rf"art[íi]culo\s+0*{re.escape(num)}\b",
                                     _plano[:220], re.I))
            _niega = bool(_NEGATIVAS.search(_plano[:400]))
            # Y QUE SEA TEXTO DE NORMA, no una respuesta sobre ella. Por lista
            # negra y mirando 400 caracteres se coló en la revisión fiscal
            # 2/2026 una nota al pie que decía «El **artículo 150** que
            # corresponde al… no puedo citarlo textualmente» —la negativa
            # estaba en la posición 464, con asteriscos—. La prueba ahora es
            # positiva: sin Markdown, sin asistente hablando, sin describirse.
            import texto_normativo as _tn
            _norma_ok, _por_que_no = _tn.es_texto_normativo(texto)
            if (not texto or "NO LOCALIZADO" in texto.upper()[:400]
                    or _corto or _colgado or _niega or not _nombra or not _norma_ok):
                _porque = ("sin texto" if not texto
                           else "el buscador dice que no lo tiene" if _niega
                           else f"no es texto de norma: {_por_que_no}" if not _norma_ok
                           else "no transcribe el artículo" if not _nombra
                           else "texto incompleto")
                print(f"   🌐 artículo {num} de «{ley[:40]}» · vuelta {i}: {_porque}")
                continue

            crudas = c["crudas"]

            # LA CITA TIENE QUE SER DE ESTA NORMA, no una cualquiera que
            # resulte oficial. Bastaba con que UNA de las citas viniera de un
            # dominio de gobierno —aunque fuera de otro asunto— para dar por
            # buena la transcripción. Se exige además que el título o la URL
            # nombren el ordenamiento: si ninguna cita lo hace, se prefiere el
            # hueco declarado.
            _voces = [w for w in re.findall(r"[\wáéíóúñ]{4,}", ley.lower())
                      if w not in ("para", "sobre", "ante", "este", "esta")]
            for titulo, url in crudas:
                dom = _dominio(url)
                if not (dom and _es_oficial(dom, AGENTE_ARTICULO["cotos"])):
                    continue
                _donde = f"{titulo} {url}".lower()
                _nombra_norma = sum(1 for w in _voces if w in _donde) >= max(
                    1, min(2, len(_voces)))
                if not _nombra_norma:
                    continue
                print(f"   🌐 artículo {num} de «{ley[:40]}» traído de "
                      f"{dom} (vuelta {i})")
                return {"texto": texto[:4000], "url": url, "dominio": dom,
                        "titulo": (titulo or dom)[:140]}

            # SIN FUENTE OFICIAL NO HAY PRECEPTO. Sonar contesta igual desde un
            # blog, y un artículo de ley sacado de un blog en un proyecto de
            # sentencia es peor que el hueco.
            print(f"   🌐 artículo {num} de «{ley[:40]}» · vuelta {i}: "
                  f"{len(crudas)} citas, ninguna oficial")

        except Exception as e:
            print(f"   🌐 artículo {num} de «{ley[:40]}» · vuelta {i} falló "
                  f"({type(e).__name__}: {str(e)[:80]})")

    return vacio


def lanzar_agentes(consulta, estado: Optional[str] = None,
                   agentes: Optional[tuple] = None) -> List[asyncio.Task]:
    """
    Lanza los agentes y devuelve sus TAREAS, sin esperarlas.

    Es la pieza que permite las fuentes EN VIVO: quien consume va recogiendo
    cada tarea conforme termina (FIRST_COMPLETED) y emite el marcador
    actualizado al frontend, en vez de esperar a que acabe la última.

    `consulta` puede ser texto o una tarea que lo entrega (la pregunta con su
    hilo); `agentes` elige cuáles corren —por omisión, los tres del globo—.
    """
    if not WEB_ACTIVA:
        return []
    if isinstance(consulta, str) and not consulta.strip():
        return []
    return [asyncio.create_task(_un_agente(a, consulta, estado)) for a in (agentes or AGENTES)]


async def buscar_en_web(consulta: str, estado: Optional[str] = None) -> Dict[str, Any]:
    """Envoltorio clásico: lanza los agentes y espera a todos. El camino
    nuevo (fuentes en vivo) usa lanzar_agentes() + fusionar() directamente."""
    tareas = lanzar_agentes(consulta, estado)
    if not tareas:
        return {"resumen": "", "fuentes": [], "agentes": [], "corrio": False}
    import web_openai
    plazo = WEB_TIMEOUT_OPENAI if web_openai.usar_openai() else WEB_TIMEOUT
    hechas, pendientes = await asyncio.wait(tareas, timeout=plazo + 2)
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
        etiqueta = next((a["etiqueta"] for a in AGENTES + (AGENTE_ACERVO_LOCAL,)
                         if a["id"] == r["id"]), r["id"])
        partes.append(f"[{etiqueta}] {r['resumen']}")
        fuentes.extend(r["fuentes"])
        aportaron.append(r["id"])
    return {"resumen": "\n\n".join(partes), "fuentes": fuentes[:6],
            "agentes": aportaron, "corrio": True}


def bloque_para_prompt(web: Dict[str, Any], entidad_debil: Optional[str] = None) -> str:
    """
    Convierte el resultado en un bloque para el prompt, con su jerarquía
    escrita de forma explícita: el modelo tiene que saber que esto NO es ley.

    `entidad_debil`: la búsqueda no la pidió el abogado, la lanzó Iurexia
    porque el acervo de esa entidad está incompleto. Cambia el motivo y
    endurece la marca: cada dato de aquí se cita como fuente de internet.
    """
    if not web or not web.get("resumen"):
        return ""

    if entidad_debil:
        cabecera = [
            "<contexto_web>",
            f"INSTRUCCIONES PARA USAR LO QUE SIGUE: el acervo de Iurexia para {entidad_debil}",
            "está incompleto, así que se buscó en internet, en sitios oficiales de la",
            "entidad, la legislación estatal que falta.",
            "",
            "1. ÚSALO para cubrir lo que el acervo no tiene, y MARCA CADA DATO que tomes",
            "   de aquí como de internet, en la misma frase: «… (fuente: internet —",
            "   <dominio>)». Nunca le pongas [Doc ID]: no viene del acervo verificado.",
            "   Esto manda sobre la regla de no citar lo que no esté en el acervo: esa",
            "   regla prohíbe citar DE MEMORIA; aquí el texto viene de un sitio oficial.",
        ]
    else:
        cabecera = [
            "<contexto_web>",
            "INSTRUCCIONES PARA USAR LO QUE SIGUE (búsqueda en internet, dominios",
            "oficiales). El usuario PIDIÓ estas fuentes con un clic: espera verlas",
            "reflejadas en tu respuesta.",
            "",
            "1. INTEGRA lo relevante en el cuerpo de la respuesta, señalándolo:",
            "   «según información en línea de <dominio>, …». Que se distinga qué",
            "   viene de internet y qué del acervo documental.",
        ]
    jerarquia = [
        "2. JERARQUÍA: si el acervo trae el mismo artículo, MANDA EL DEL ACERVO. Lo",
        "   de internet sólo cubre lo que falta, y al usarlo recomienda cotejar su",
        "   vigencia en la fuente oficial.",
    ] if entidad_debil else [
        "2. JERARQUÍA: esto NO es la ley y NO sustituye a los artículos del",
        "   contexto documental. Si contradice un artículo, MANDA EL ARTÍCULO,",
        "   y puedes señalar que hay información en línea que apunta a un",
        "   cambio reciente. El fundamento jurídico son los artículos y las",
        "   tesis verificadas, nunca una página web.",
    ]
    lineas = cabecera + jerarquia + [
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
    vistos = set()
    mostradas = 0
    for f in (fuentes or []):
        if mostradas >= maximo:
            break
        dominio = str(f.get("dominio", "") or "")
        url = str(f.get("url", "") or "")
        if not url:
            continue
        titulo = str(f.get("titulo") or dominio)
        # Los buscadores devuelven el título con la etiqueta del formato
        # pegada («[DOC] LEY DE ADQUISICIONES…», «(PDF) …»). Estorba y ya se
        # ve en la URL.
        titulo = re.sub(r"^\s*[\[\(](?:PDF|DOC|DOCX|XLS|PPT)[\]\)]\s*", "", titulo, flags=re.I)
        titulo = " ".join(titulo.split())[:110] or dominio
        # Un mismo documento oficial aparece en varias URLs del mismo portal
        # (visores, descargas, versiones). Con el título ya limpio, se veían
        # dos filas idénticas seguidas.
        clave = (dominio.lower(), titulo.lower())
        if clave in vistos:
            continue
        vistos.add(clave)
        mostradas += 1
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
