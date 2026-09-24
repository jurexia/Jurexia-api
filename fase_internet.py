"""LA LÍNEA DE LA CORTE, BUSCADA EN INTERNET Y COMPROBADA EN EL ACERVO.

═══ POR QUÉ EXISTE (23-sep-2026) ══════════════════════════════════════════
David, sobre el amparo en revisión 711/2025 —bloqueo de cuentas por la Unidad
de Inteligencia Financiera—: «hay una extensa línea jurisprudencial y de
resoluciones de la SCJN que evolucionó el tratamiento a los bloqueos de
cuentas. Si buscas en internet verás que hay precedentes sumamente relevantes
que no se toman en cuenta en el proyecto. Ya teníamos la capacidad de buscar
en internet, pero en este caso no se hizo».

Tenía razón en las dos cosas. La capa web existía y SÓLO la usaba el chat; el
taller buscaba el acervo por parecido de rubro y por co-citación, que son
buenos para lo que ya está indexado y ciegos para lo que la Corte resolvió el
mes pasado o para una línea que se lee mejor en un comunicado que en un rubro.
Sobre el bloqueo de cuentas la Corte fue de «inconstitucional» a
«constitucional cuando cumple compromisos internacionales» a los matices
sobre personas morales vinculadas, y ese recorrido no lo trae ninguna
búsqueda semántica del problema.

═══ LA REGLA QUE NO SE NEGOCIA ════════════════════════════════════════════
Lo que internet dice NO se cita. Lo que internet dice y EL ACERVO CONFIRMA, sí.

  1. Se pregunta al buscador —gpt-6-luna con la búsqueda web de OpenAI desde
     el 23-sep-2026; sonar si BUSQUEDA_WEB_MOTOR=openrouter— por la línea de
     precedentes sobre el problema (ver web_openai.py: por qué y cuánto).
  2. De la respuesta se extraen los REGISTROS DIGITALES —siete dígitos— y las
     claves de tesis.
  3. Cada registro se busca en `jurisprudencia_nacional_v3`. El que está,
     entra al material con su texto íntegro y marcado `de_internet=True`: el
     estudio lo cita como cualquier otra tesis, porque LO ES.
  4. El que no está en el acervo NO ENTRA COMO TESIS. Queda como pista con
     su URL, para que quien firma la busque; y al modelo se le dice que de
     esa lista no se cita ningún registro.

Un registro que sonar escribió mal —le pasa— no puede acabar en una
sentencia. Con este orden, no puede: el acervo es la única puerta.

═══ QUÉ CUESTA ═════════════════════════════════════════════════════════════
Tres llamadas en paralelo a gpt-6-luna (medido: 21 s de mediana por ángulo,
42 s la peor; 0,14 USD el problema) y un scroll a Qdrant por registro. Se
apaga con BUSQUEDA_WEB_ACTIVA=false como el resto de la capa web.
"""
from __future__ import annotations

import asyncio
import os
import re
from typing import Any, Dict, List, Optional

# ── Dominios en los que la Corte publica lo suyo ──────────────────────────
OFICIALES_CORTE = (
    "sjf.scjn.gob.mx", "sjf2.scjn.gob.mx", "scjn.gob.mx", "www.scjn.gob.mx",
    "bj.scjn.gob.mx", "sitios.scjn.gob.mx", "www2.scjn.gob.mx",
    "cjf.gob.mx", "www.cjf.gob.mx", "dof.gob.mx", "www.dof.gob.mx",
)

RX_REGISTRO = re.compile(r"\b(2[0-9]{6}|1[0-9]{5,6}|[1-9][0-9]{5})\b")
RX_CLAVE = re.compile(
    r"(?:(?:1a|2a|P|PC)\.?\s*/\s*J\.?\s*\d{1,4}/\d{4}(?:\s*\(\d{1,2}a\.\))?)"
    r"|(?:(?:1a|2a|P)\.\s*[IVXLC]+/\d{4}(?:\s*\(\d{1,2}a\.\))?)")

TOPE_REGISTROS = 12
TIMEOUT_S = float(os.getenv("BUSQUEDA_WEB_TIMEOUT", "18"))
# gpt-6-luna tarda más que sonar y trae más: 21 s de mediana por ángulo y 42 s
# el peor de 36 medidos. Con los 18 s de sonar se cortaría a media búsqueda.
TIMEOUT_OPENAI_S = float(os.getenv("BUSQUEDA_WEB_OPENAI_TIMEOUT", "60"))


def _dominio(url: str) -> str:
    m = re.match(r"https?://([^/]+)", url or "")
    return (m.group(1).lower() if m else "").removeprefix("www.")


def _es_de_la_corte(url: str) -> bool:
    d = _dominio(url)
    return any(d == c.removeprefix("www.") or d.endswith("." + c.removeprefix("www."))
               for c in OFICIALES_CORTE)


# ═══ TRES ÁNGULOS, NO UNA PREGUNTA ═════════════════════════════════════════
# Medido el 23-sep-2026 con el problema del bloqueo de cuentas: una sola
# pregunta general devolvió UN registro y cuatro «no localizado». La línea
# existe —2a./J. 46/2018, 2a./J. 47/2018, la acción de inconstitucionalidad
# 58/2022— pero sonar la encuentra cuando se le pregunta por partes: la
# jurisprudencia por contradicción, la naturaleza del acto, y el caso concreto.
ANGULOS = (
    "la JURISPRUDENCIA de la Suprema Corte (Pleno o Salas, por contradicción de "
    "tesis o reiteración) que fija el criterio, con su registro digital y su clave",
    "la NATURALEZA JURÍDICA del acto según la Corte —acto de molestia o de "
    "privación, medida cautelar administrativa, si exige orden judicial previa o "
    "garantía de audiencia previa— con registro digital de cada tesis",
    "los PRECEDENTES MÁS RECIENTES (últimos tres años) de la Suprema Corte, los "
    "Plenos Regionales y los Tribunales Colegiados sobre el supuesto de hecho "
    "concreto, con registro digital, y las acciones de inconstitucionalidad o "
    "contradicciones que lo resolvieron",
)


def prompt_busqueda(problema: str, hechos: str, tipo_asunto: str, angulo: str = "") -> str:
    return (
        "Eres investigador de un Tribunal Colegiado mexicano. Busca en el "
        "Semanario Judicial de la Federación (sjf2.scjn.gob.mx), en scjn.gob.mx "
        "y en bj.scjn.gob.mx la LÍNEA DE PRECEDENTES de la Suprema Corte y de "
        "los Plenos Regionales sobre este problema jurídico:\n\n"
        f"PROBLEMA: {problema}\n"
        + (f"HECHOS RELEVANTES: {hechos[:1200]}\n" if hechos else "")
        + (f"\nBUSCA EN CONCRETO: {angulo}\n" if angulo else "")
        + "\nQUIERO, en orden cronológico y con cada criterio en su renglón:\n"
        "- El REGISTRO DIGITAL (número de siete dígitos del Semanario) de cada "
        "tesis o jurisprudencia, SIEMPRE que lo tengas. Es el dato más "
        "importante.\n"
        "- La clave de la tesis (por ejemplo 2a./J. 46/2018 (10a.)) y su rubro.\n"
        "- Qué resolvió y cómo cambió el criterio anterior, en una frase.\n"
        "- Los amparos en revisión o contradicciones de la SCJN que fijaron el "
        "criterio, con su número.\n\n"
        "No inventes registros: si no lo sabes, escribe «registro: no localizado». "
        "Responde en español, sin introducción."
    )


async def buscar_linea(problema: str, hechos: str = "", tipo_asunto: str = "") -> Dict[str, Any]:
    """Pregunta a internet por la línea de la Corte, desde tres ángulos a la
    vez, y fusiona. Nunca lanza."""
    vacio = {"texto": "", "registros": [], "claves": [], "fuentes": []}
    import web_openai
    activa = os.getenv("BUSQUEDA_WEB_ACTIVA", "false").lower() in ("1", "true", "si", "sí")
    if not activa or not web_openai.hay_motor() or not (problema or "").strip():
        return vacio
    partes = await asyncio.gather(*[_un_angulo(problema, hechos, tipo_asunto, a) for a in ANGULOS])
    texto = "\n\n".join(p["texto"] for p in partes if p["texto"])
    registros = list(dict.fromkeys(r for p in partes for r in p["registros"]))
    claves = list(dict.fromkeys(c for p in partes for c in p["claves"]))
    fuentes, vistos = [], set()
    for p in partes:
        for f in p["fuentes"]:
            if f["url"] not in vistos:
                vistos.add(f["url"]); fuentes.append(f)
    print(f"   🌐 línea de la Corte (3 ángulos): {len(texto)} car. · {len(registros)} registros "
          f"· {len(claves)} claves · {len(fuentes)} fuentes oficiales")
    return {"texto": texto, "registros": registros[:TOPE_REGISTROS * 2],
            "claves": claves[:TOPE_REGISTROS * 2], "fuentes": fuentes[:8]}


async def _un_angulo(problema: str, hechos: str, tipo_asunto: str, angulo: str) -> Dict[str, Any]:
    import web_openai
    if web_openai.usar_openai():
        return await _un_angulo_openai(problema, hechos, tipo_asunto, angulo)
    vacio = {"texto": "", "registros": [], "claves": [], "fuentes": []}
    clave = os.getenv("OPENROUTER_API_KEY", "")
    try:
        import httpx
        modelo = os.getenv("BUSQUEDA_WEB_MODELO", "perplexity/sonar")
        async with httpx.AsyncClient(timeout=TIMEOUT_S) as cli:
            r = await cli.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {clave}"},
                json={"model": modelo,
                      "messages": [{"role": "user",
                                    "content": prompt_busqueda(problema, hechos, tipo_asunto, angulo)}],
                      "max_tokens": 1400})
            r.raise_for_status()
            d = r.json()
        msg = (d.get("choices") or [{}])[0].get("message", {}) or {}
        texto = (msg.get("content") or "").strip()
        crudas: List[tuple] = []
        for a in (msg.get("annotations") or []):
            uc = a.get("url_citation") or {}
            if uc.get("url"):
                crudas.append((uc.get("title") or "", uc["url"]))
        for u in (d.get("citations") or []):
            crudas.append(("", u))
        fuentes, vistos = [], set()
        for titulo, url in crudas:
            if url in vistos or not _es_de_la_corte(url):
                continue
            vistos.add(url)
            fuentes.append({"titulo": (titulo or _dominio(url))[:140], "url": url})
        registros = []
        for m in RX_REGISTRO.finditer(texto):
            reg = m.group(1)
            # Los registros del Semanario van de 6 a 7 dígitos; un año (2018)
            # o un número de expediente (711/2025) no lo son. Se exige que no
            # vaya pegado a una barra ni sea un año plausible de cuatro.
            if len(reg) >= 6 and reg not in registros:
                registros.append(reg)
        claves = list(dict.fromkeys(" ".join(c.split()) for c in RX_CLAVE.findall(texto)))
        return {"texto": texto, "registros": registros, "claves": claves, "fuentes": fuentes[:6]}
    except Exception as e:
        print(f"   🌐 un ángulo de la línea falló ({type(e).__name__}: {str(e)[:80]})")
        return vacio


def _registros_del_texto(texto: str) -> List[str]:
    registros = []
    for m in RX_REGISTRO.finditer(texto):
        reg = m.group(1)
        if len(reg) >= 6 and reg not in registros:
            registros.append(reg)
    return registros


async def _un_angulo_openai(problema: str, hechos: str, tipo_asunto: str, angulo: str) -> Dict[str, Any]:
    """El mismo ángulo con gpt-6-luna. Con la configuración que se midió:
    búsqueda abierta —el cotejo contra el acervo filtra después—, página
    completa (contexto high) y razonamiento bajo."""
    import web_openai
    vacio = {"texto": "", "registros": [], "claves": [], "fuentes": []}
    r = await web_openai.buscar(prompt_busqueda(problema, hechos, tipo_asunto, angulo),
                                contexto="high", esfuerzo="low", timeout=TIMEOUT_OPENAI_S)
    if r["error"] or not r["texto"]:
        print(f"   🌐 un ángulo de la línea falló ({r['error'] or 'sin texto'}, {r['segundos']} s)")
        return vacio
    # Sin enlaces ANTES de buscar números: el nombre de un PDF no es un registro.
    texto = web_openai.sin_enlaces(r["texto"])
    fuentes, vistos = [], set()
    for titulo, url in r["citas"] + [("", u) for u in r["consultadas"]]:
        if url in vistos or not _es_de_la_corte(url):
            continue
        vistos.add(url)
        fuentes.append({"titulo": (titulo or _dominio(url))[:140], "url": url})
    claves = list(dict.fromkeys(" ".join(c.split()) for c in RX_CLAVE.findall(texto)))
    return {"texto": texto, "registros": _registros_del_texto(texto), "claves": claves,
            "fuentes": fuentes[:6]}


# ═══ EL NÚMERO NO BASTA: SE COTEJA EL RUBRO ════════════════════════════════
# Medido el 23-sep-2026 en la primera corrida real: sonar escribió «registro
# 2016904 — LISTA DE PERSONAS BLOQUEADAS. LA SUSPENSIÓN PROVISIONAL…» y
# «2016905 — …PERSONA MORAL…». Los dos registros EXISTEN en el acervo: 2016904
# es «AGRAVIOS INOPERANTES POR INSUFICIENTES EN EL RECURSO DE REVISIÓN FISCAL»
# y 2016905 es «ALIMENTOS EN CASOS DE DIVORCIO». Sonar acertó el 2016903 y
# después contó de uno en uno. Es la serie correlativa que ya delató a los
# registros inventados en el chat (ver `registros_fuera_del_contexto`): el
# espacio de registros es tan denso que casi cualquier número plausible es
# ALGUNA tesis real, así que existir no prueba nada.
#
# Lo que prueba es que el rubro que sonar atribuye al número coincida con el
# rubro que el acervo tiene para ese número. Si no coincide, ese registro no
# entra: es exactamente la cita que resiste el primer clic y se cae en el
# tribunal.
_PALABRAS_VACIAS = {"de", "del", "la", "el", "los", "las", "en", "y", "o", "a", "que",
                    "por", "para", "con", "su", "sus", "se", "es", "un", "una", "al",
                    "no", "lo", "como", "sin", "sobre", "ante", "cuando", "ser"}


def _fichas(t: str) -> set:
    import unicodedata as _u
    t = _u.normalize("NFD", (t or "").lower())
    t = "".join(c for c in t if _u.category(c) != "Mn")
    return {w for w in re.findall(r"[a-z0-9]{3,}", t) if w not in _PALABRAS_VACIAS}


def _rubro_junto_a(texto: str, ancla: str) -> str:
    """El rubro que sonar escribió junto a un registro o una clave: lo que va
    en negritas o en versales en los 260 caracteres que siguen."""
    i = texto.find(ancla)
    if i < 0:
        return ""
    trozo = texto[i:i + 320]
    m = re.search(r"\*\*([^*]{12,220})\*\*", trozo)
    if m:
        return m.group(1)
    m = re.search(r"([A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ ,.;:()\-]{20,220})", trozo)
    return m.group(1) if m else ""


# LAS PALABRAS QUE HACEN AL TEMA, no las que aparecen en cualquier tesis.
# Medido en la segunda corrida: con «una palabra en común» pasaron una tesis
# de aviación civil y tres de arresto contra persona moral, porque «persona»,
# «moral» y «apoderado» están en el problema y en media colección. El tema se
# reconoce por sus términos distintivos: los que no son de derecho en general
# —«amparo», «juicio», «persona», «acto»— sino de ESTE problema.
_GENERICAS = _PALABRAS_VACIAS | {
    "amparo", "juicio", "acto", "actos", "persona", "personas", "moral", "morales",
    "fisica", "fisicas", "autoridad", "autoridades", "procede", "procedencia",
    "improcedente", "improcedencia", "recurso", "revision", "ley", "articulo",
    "articulos", "constitucion", "constitucional", "federal", "derecho", "derechos",
    "reclamado", "reclamada", "efectos", "efecto", "caso", "cuando", "contra",
    "indirecto", "directo", "suspension", "provisional", "definitiva", "sentencia",
    "tribunal", "sala", "pleno", "circuito", "materia", "apoderado", "apoderada",
    "representante", "autorizado", "autorizada", "titular", "medida", "medidas",
    "puede", "pueden", "debe", "deben", "tiene", "tienen", "extender", "extenderse",
    "inclusion", "incluido", "incluida", "relacion", "vinculacion", "recursos",
}


# LA RAÍZ, NO LA PALABRA. «bloqueo», «bloqueadas» y «bloquear» son el mismo
# tema, y «bancaria» y «bancarias» también. Se recorta a seis letras, que es
# tosco y basta; y «congelamiento» —que es como llamaba la Corte al bloqueo
# antes de 2018— se lleva a la misma raíz a mano.
_SINONIMOS = {"congel": "bloque", "inmovi": "bloque", "asegur": "bloque"}


def _raiz(w: str) -> str:
    r = w[:6] if len(w) > 6 else w
    return _SINONIMOS.get(r, r)


def _distintivas(texto: str) -> set:
    return {_raiz(w) for w in _fichas(texto) if w not in _GENERICAS}


def _corresponde(rubro_sonar: str, rubro_acervo: str, tema: str) -> bool:
    """¿El rubro que dijo internet es el que el acervo tiene para ese número,
    y además va del tema? Dos filtros, y los dos hacen falta."""
    a, b = _fichas(rubro_sonar), _fichas(rubro_acervo)
    if a and b:
        comunes = len(a & b) / max(1, min(len(a), len(b)))
        if comunes < 0.45:
            return False
    return _del_tema(rubro_acervo, tema)


def _del_tema(rubro_acervo: str, tema: str) -> bool:
    """El rubro comparte al menos DOS términos distintivos con el problema, o
    uno si el problema es corto. «Bloqueo», «cuentas», «bancarias», «lista»,
    «bloqueadas», «inteligencia», «financiera», «115», «lavado»: ésos
    distinguen. «Persona moral» no."""
    t = _distintivas(tema)
    if not t:
        return True
    comunes = _distintivas(rubro_acervo) & t
    return len(comunes) >= (2 if len(t) >= 6 else 1)


async def precedentes_verificados(qdrant, problema: str, hechos: str = "",
                                  tipo_asunto: str = "", embed_juris=None) -> Dict[str, Any]:
    """Lo que internet dice Y el acervo confirma, listo para el material.

    Devuelve:
      tesis      → dicts de tesis del acervo (con texto), marcadas de_internet
      pistas     → lo que internet citó y el acervo NO tiene (sólo para el aviso)
      resumen    → el texto de sonar, para orientar al modelo (NO para citar)
      fuentes    → URLs oficiales
    """
    linea = await buscar_linea(problema, hechos, tipo_asunto)
    if not linea["texto"]:
        return {"tesis": [], "pistas": [], "resumen": "", "fuentes": [], "buscado": False}
    tesis: List[dict] = []
    if qdrant is not None and (linea["registros"] or linea["claves"]):
        try:
            import fase6_rag as _rag
            if linea["registros"]:
                tesis = await _rag.tesis_por_registro(qdrant, linea["registros"])
            # POR CLAVE TAMBIÉN. Sonar sabe la clave —«2a./J. 46/2018 (10a.)»—
            # muchas más veces que el registro; la clave resuelve contra el
            # acervo con las variantes ortográficas que ya tolera `_ficha_de_cita`.
            vistos = {t.get("registro") for t in tesis}
            for c in linea["claves"][:TOPE_REGISTROS]:
                f = await _rag._ficha_de_cita(qdrant, c)
                if f and f.get("registro") and f["registro"] not in vistos and f.get("rubro"):
                    vistos.add(f["registro"]); f["tecnica"] = True; tesis.append(f)
        except Exception as e:
            print(f"   ⚠️ no se pudieron comprobar los registros de internet: {e}")
    # EL COTEJO. Cada tesis que el acervo devolvió se enfrenta al rubro que
    # sonar le atribuyó y al tema. Lo que no cuadra sale de la lista y pasa a
    # «pistas», que es donde va lo que no se puede citar.
    tema = f"{problema} {hechos}"
    buenas, descartadas = [], []
    for t in tesis:
        ancla = str(t.get("registro") or "")
        rubro_sonar = _rubro_junto_a(linea["texto"], ancla) if ancla else ""
        if not rubro_sonar:
            for c in linea["claves"]:
                if _fichas(c) and _fichas(c) <= _fichas(str(t.get("clave") or t.get("clave_tesis") or "")):
                    rubro_sonar = _rubro_junto_a(linea["texto"], c)
                    break
        if _corresponde(rubro_sonar, str(t.get("rubro") or ""), tema):
            buenas.append(t)
        else:
            descartadas.append(f"{ancla} ({str(t.get('rubro') or '')[:50]}…)")
    if descartadas:
        print(f"   🌐 descartadas por rubro o tema: {descartadas}")
    tesis = buenas
    confirmados = {t.get("registro") for t in tesis}
    for t in tesis:
        t["de_internet"] = True
        t["tecnica"] = True          # exentas del recorte del prompt, como las de técnica
    # Y la misma línea, pedida al acervo con las frases de sonar como llave.
    _mas = await _linea_desde_el_acervo(qdrant, embed_juris, linea["texto"], tema, confirmados)
    if _mas:
        print(f"   🌐 la línea, desde el acervo por tema: {len(_mas)} más → "
              f"{[t.get('registro') for t in _mas]}")
        tesis = tesis + _mas
    confirmados = {t.get("registro") for t in tesis}
    pistas = [r for r in linea["registros"] if r not in confirmados]
    print(f"   🌐 precedentes de internet: {len(tesis)} confirmados en el acervo · "
          f"{len(pistas)} sin confirmar → {pistas[:6]}")
    return {"tesis": tesis, "pistas": pistas, "resumen": linea["texto"],
            "fuentes": linea["fuentes"], "buscado": True}


# ═══ LA LÍNEA TAMBIÉN SE PIDE AL ACERVO POR SU TEMA ════════════════════════
# Medido el 23-sep-2026 sobre el 711/2025: el acervo tiene DIEZ criterios de
# la línea de bloqueos —2029503 y 2029549 sobre la «petición expresa» de 2024,
# 2030171 sobre la persona moral cuyo apoderado está listado, 2023428, 2030583,
# 2026616…— y al material llegaron cuatro. La búsqueda del taller se hace con
# la PREGUNTA del problema, y una pregunta de veinte palabras no se parece al
# rubro de una tesis aunque sea la que decide el punto.
#
# Aquí se busca con las frases que sonar devolvió —rubros y descripciones de
# la línea— que SÍ se parecen a los rubros del acervo, y se filtra por tema
# igual que lo de internet. Es la misma verdad con otra llave.
async def _linea_desde_el_acervo(qdrant, embed_juris, resumen: str, tema: str,
                                 ya: set, tope: int = 8) -> List[dict]:
    if qdrant is None or embed_juris is None or not (resumen or "").strip():
        return []
    try:
        import fase6_rag as _rag
        # Las frases de la línea: cada renglón con algo de sustancia.
        frases = [ln.strip("-•* ") for ln in resumen.splitlines()
                  if len(ln.strip()) > 40 and not ln.strip().lower().startswith("registro:")]
        frases = frases[:6]
        if not frases:
            return []
        fuera, vistos = [], set(ya)
        tema_f = _fichas(tema)
        for fr in frases:
            v = await embed_juris(fr[:600])
            r = await _rag._buscar(qdrant, _rag.COLECCION_JURIS, "rubro", v, tope)
            for pl in (r or []):
                t = _rag._tesis_de(pl)        # la misma forma que el resto del material
                reg = t.get("registro")
                if not reg or reg in vistos or not t.get("rubro"):
                    continue
                # Del tema, de verdad: dos términos distintivos en común con el
                # problema. Sin esto la semántica trae vecinos de barrio.
                if not _del_tema(t["rubro"], tema):
                    continue
                vistos.add(reg); t["tecnica"] = True; t["de_internet"] = True
                fuera.append(t)
                if len(fuera) >= tope:
                    return fuera
        return fuera
    except Exception as e:
        print(f"   🌐 la línea desde el acervo falló: {type(e).__name__}: {str(e)[:100]}")
        return []


def bloque_para_prompt(resultado: Dict[str, Any]) -> str:
    """Cómo se le cuenta al modelo. Las tesis confirmadas ya van en el
    material como cualquier otra; aquí sólo va el CONTEXTO de la línea y la
    prohibición sobre lo no confirmado."""
    if not resultado or not resultado.get("buscado"):
        return ""
    partes = ["", "═" * 71, "LA LÍNEA DE LA CORTE SOBRE ESTE PROBLEMA, BUSCADA EN INTERNET",
              "═" * 71]
    n = len(resultado.get("tesis") or [])
    if n:
        regs = ", ".join(str(t.get("registro")) for t in resultado["tesis"][:TOPE_REGISTROS])
        partes.append(
            f"Se localizaron {n} criterios de esa línea y están CONFIRMADOS en el "
            f"acervo: van entre las tesis de abajo, marcados como tales, con su texto "
            f"íntegro. Registros: {regs}. Úsalos: son la evolución del criterio y un "
            f"estudio que la ignora está desactualizado.")
    resumen = (resultado.get("resumen") or "").strip()
    if resumen:
        partes.append("CÓMO EVOLUCIONÓ EL CRITERIO, según la búsqueda (orientación, no cita):\n"
                      + resumen[:2500])
    pistas = resultado.get("pistas") or []
    if pistas:
        partes.append(
            "REGISTROS QUE LA BÚSQUEDA MENCIONÓ Y EL ACERVO NO CONFIRMA: "
            + ", ".join(pistas[:10])
            + ". NO LOS CITES. Un registro que no está en el acervo no se puede "
              "comprobar, y una cita que no se puede comprobar no entra en un "
              "proyecto. Quedan en el apartado ADVERTENCIAS para quien firma.")
    return "\n".join(partes) + "\n"


def aviso(resultado: Dict[str, Any]) -> str:
    if not resultado or not resultado.get("buscado"):
        return ""
    n = len(resultado.get("tesis") or [])
    pistas = resultado.get("pistas") or []
    urls = [f.get("url") for f in (resultado.get("fuentes") or []) if f.get("url")]
    s = (f"SE BUSCÓ EN INTERNET LA LÍNEA DE LA CORTE: {n} criterio(s) localizados y "
         f"confirmados en el acervo entraron al estudio.")
    if pistas:
        s += (f" La búsqueda mencionó además {len(pistas)} registro(s) que el acervo no "
              f"tiene ({', '.join(pistas[:6])}): no se citaron; compruébalos en el "
              f"Semanario antes de usarlos.")
    if urls:
        s += " Fuentes: " + " · ".join(urls[:3])
    return s
