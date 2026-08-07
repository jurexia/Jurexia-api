"""
La capa doctrinal de Iurexia: obras jurídicas de referencia, sólo-cita.

EL CONTRATO (decisión de David, 7-ago-2026)
-------------------------------------------
El texto de las obras vive en Qdrant únicamente para que el motor recupere y
el modelo entienda. Al abogado se le sirve la CITA: autor, obra, año, página
y enlace al PDF en su fuente oficial. Nunca el texto corrido. Es el derecho
de cita del art. 148 fr. I de la LFDA hecho arquitectura, y por eso esta capa
NO entra al sistema de [Doc ID] ni al visor /cita del acervo: sus fuentes se
muestran en su propia tarjeta, con el enlace apuntando fuera.

CUÁNDO ENTRA LA DOCTRINA (regla medida, no supuesta)
----------------------------------------------------
El umbral solo no separa: «plazo para contestar la demanda» puntúa 0.545 y
«qué es el control de convencionalidad» 0.539 — medido el 7-ago-2026. La
regla es doble:

  · score >= 0.60 — el tema pega fuerte, entra siempre; o
  · score >= 0.50 Y la consulta es conceptual («qué es», «concepto»,
    «naturaleza jurídica», «doctrina», «teoría de»...).

Con las nueve consultas de calibración, clasifica 9/9: las conceptuales
entran, «requisitos del divorcio» (0.430) y «multa por no verificar» (0.458)
quedan fuera. La doctrina ilustra conceptos; no estorba trámites.
"""
from __future__ import annotations

import os
import re
import unicodedata
from typing import Any, Dict, List, Optional

COLECCION = "doctrina"
UMBRAL_FUERTE = 0.60
UMBRAL_CONCEPTUAL = 0.50
MAX_FRAGMENTOS = 3

_RE_CONCEPTUAL = re.compile(
    r'\b(qu[ée] es|concepto|definici[óo]n|naturaleza jur[íi]dica|doctrina'
    r'|doctrinal|te[oó]r[íi]a de|seg[úu]n la teor[íi]a|principio de'
    r'|fundamento te[óo]rico|qu[ée] se entiende por)\b', re.I)


def activa() -> bool:
    return os.getenv("DOCTRINA_ACTIVA", "true").lower() != "false"


def es_conceptual(consulta: str) -> bool:
    return bool(_RE_CONCEPTUAL.search(consulta or ""))


async def buscar(qdrant_client, dense_vector, consulta: str) -> List[Dict[str, Any]]:
    """Fragmentos doctrinales que superan la regla de entrada.

    Falla en silencio a lista vacía: si la colección no existe o el clúster
    tose, la consulta del abogado sigue exactamente igual que antes de que
    esta capa existiera.
    """
    if not activa():
        return []
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        res = await qdrant_client.query_points(
            collection_name=COLECCION,
            query=dense_vector,
            using="dense",
            limit=MAX_FRAGMENTOS,
            query_filter=Filter(must=[
                FieldCondition(key="subtipo", match=MatchValue(value="doctrina")),
            ]),
            with_payload=True,
        )
        puntos = getattr(res, "points", res) or []
    except Exception as e:
        print(f"   📚 Doctrina no disponible ({type(e).__name__}); la consulta sigue sin ella")
        return []

    minimo = UMBRAL_CONCEPTUAL if es_conceptual(consulta) else UMBRAL_FUERTE
    frags = []
    for p in puntos:
        if (p.score or 0) < minimo:
            continue
        pl = p.payload or {}
        frags.append({
            "id": str(p.id),
            "score": p.score,
            "texto": pl.get("texto", ""),
            "autor": pl.get("autor", ""),
            "obra": pl.get("obra", ""),
            "anio": pl.get("anio"),
            "editorial": pl.get("editorial"),
            "pagina": pl.get("pagina_impresa") or pl.get("pagina_pdf"),
            "pagina_pdf": pl.get("pagina_pdf"),
            "url_oficial": pl.get("url_oficial", ""),
        })
    return frags


def bloque_para_prompt(frags: List[Dict[str, Any]]) -> str:
    """Lo que ve el MODELO. Con la regla de uso pegada al material."""
    if not frags:
        return ""
    lineas = [
        "<doctrina>",
        "Fragmentos de obras jurídicas de referencia, con su cita. Reglas:",
        "1. ÚSALOS para enriquecer el concepto, atribuyendo SIEMPRE: autor,",
        "   obra y página, p. ej. (Atienza, Las razones del derecho, p. 45).",
        "2. Si citas textual, MÁXIMO 40 palabras y entre comillas — es derecho",
        "   de cita, no reproducción. Sólo puedes citar textual lo que esté",
        "   AQUÍ; jamás de memoria.",
        "3. La doctrina ILUSTRA; el fundamento son la ley y la jurisprudencia.",
        "",
    ]
    for f in frags:
        lineas.append(
            f"— {f['autor']}, «{f['obra']}», p. {f['pagina']}:\n{f['texto'][:1200]}\n")
    lineas.append("</doctrina>")
    return "\n".join(lineas)


def _esc(v: Any) -> str:
    return (str(v or "").replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def bloque_doctrina_html(frags: List[Dict[str, Any]], citas_no_verificadas: int = 0) -> str:
    """La tarjeta que ve el ABOGADO. Reutiliza las clases de la tarjeta de
    fuentes web —ya desplegadas y probadas en móvil— con su propio título:
    cero cambios de CSS que desplegar. Una sola línea, porque el formateador
    del chat convierte los saltos en <br/>."""
    if not frags:
        return ""
    # Agrupada POR OBRA, con las páginas juntas: dos fragmentos del mismo tomo
    # salían como dos filas idénticas (pp. 711 y 712) y parecía un error.
    obras: dict = {}
    for f in frags:
        if not f.get("url_oficial"):
            continue
        clave = (f["autor"], f["obra"])
        o = obras.setdefault(clave, {**f, "paginas": []})
        if f.get("pagina") and f["pagina"] not in o["paginas"]:
            o["paginas"].append(f["pagina"])

    partes = ['<div class="fuentes-web"><div class="fw-cab">'
              '<span class="fw-globo">\U0001F4DA</span>'
              '<span>Doctrina consultada</span></div><div class="fw-lista">']
    for (autor, obra), o in obras.items():
        pags = sorted(p for p in o["paginas"] if p)
        etiqueta = ("p. " + str(pags[0])) if len(pags) == 1 else ("pp. " + ", ".join(map(str, pags)))
        enlace = f"{o['url_oficial']}#page={o['pagina_pdf']}" if o.get("pagina_pdf") else o["url_oficial"]
        anio = f", {o['anio']}" if o.get("anio") else ""
        partes.append(
            f'<a class="fw-item" href="{_esc(enlace)}" target="_blank" rel="noopener noreferrer">'
            f'<span class="fw-ico fw-ico--oficial"></span>'
            f'<span class="fw-txt"><span class="fw-tit">{_esc(autor)} — {_esc(obra)}</span>'
            f'<span class="fw-dom">{_esc(o.get("editorial") or "")}{_esc(anio)} · {_esc(etiqueta)}</span></span>'
            f'<span class="fw-flecha">&#8599;</span></a>')
    nota = ("Referencias doctrinales con su página; el enlace abre la obra en su "
            "repositorio oficial. La doctrina ilustra el concepto; el fundamento "
            "jurídico son la ley y la jurisprudencia citadas arriba.")
    if citas_no_verificadas:
        nota += (f" ⚠️ {citas_no_verificadas} cita(s) textual(es) de la respuesta no "
                 "pudieron verificarse contra la obra: tómelas como paráfrasis.")
    partes.append(f'</div><div class="fw-nota">{_esc(nota)}</div></div>')
    return "".join(partes)


# ── Verificación de citas textuales ─────────────────────────────────────
# El mismo principio que el detector del Semanario: sólo se afirma lo
# comprobable. Toda cita textual que la respuesta atribuya a la doctrina se
# busca, normalizada, dentro de los fragmentos recuperados. La respuesta ya
# se transmitió (streaming): lo no verificable no se puede retirar, pero SÍ
# se advierte en la tarjeta y se cuenta en los logs.

def normalizar(t: str) -> str:
    t = unicodedata.normalize("NFD", t or "").lower()
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return " ".join(t.split())


_RE_COMILLAS = re.compile(r'[«"“]([^»"”]{60,600})[»"”]')


def citas_sin_verificar(respuesta: str, frags: List[Dict[str, Any]]) -> int:
    """Cuántas citas textuales largas atribuidas a la doctrina NO aparecen en
    los fragmentos. Sólo mira comillas de 60+ caracteres con un autor
    doctrinal mencionado cerca: las comillas cortas y las de leyes/tesis no
    son asunto de esta capa."""
    if not frags:
        return 0
    corpus = normalizar(" ".join(f["texto"] for f in frags))
    apellidos = set()
    for f in frags:
        for token in re.split(r"[,y\s]+", str(f["autor"])):
            if len(token) > 3:
                apellidos.add(normalizar(token))
    fallidas = 0
    for m in _RE_COMILLAS.finditer(respuesta or ""):
        cita = m.group(1)
        contexto = normalizar(respuesta[max(0, m.start() - 220):m.end() + 220])
        if not any(a in contexto for a in apellidos):
            continue          # no se atribuye a la doctrina: no es nuestro caso
        if normalizar(cita) not in corpus:
            fallidas += 1
    return fallidas
