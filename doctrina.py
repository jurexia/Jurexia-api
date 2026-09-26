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

(Eso cambió después: hoy la doctrina se cita con [Doc ID] como la ley, y desde
el 25-sep-2026 abre en el visor, en su página y con el pasaje resaltado —ver
«El contrato del visor» más abajo—. El texto sigue sin copiarse: el visor lee
el PDF de la UNAM por el proxy.)

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
        frags.append(fragmento(p.id, p.payload or {}, p.score))
    return frags


# ── El contrato del visor (25-sep-2026) ─────────────────────────────────
# David: «En todas estas nuevas el visor no está disponible, tenemos que
# implementarlo como todas nuestras fuentes». La doctrina llegaba al visor con
# `pdf_url` = url_oficial + «#page=N»: el proxy /api/ley/pdf la rechazaba y la
# app decía «No se pudo abrir el PDF aquí». Ahora viaja como la Corte IDH:
# `pdf_url` = `url_oficial` SIN «#page», `pagina` = la del PDF del capítulo
# (`pagina_pdf`), y `ancla`, las primeras palabras literales del trozo, para
# que el visor encuentre el pasaje en esa página y lo resalte.
#
# El PDF se lee de la Biblioteca Jurídica Virtual por el proxy, como los de
# diputados.gob.mx: NO se copia a nuestro almacenamiento (derechos de autor).
# Esto no contradice «sólo-cita»: el abogado abre la obra EN SU REPOSITORIO,
# en su página; lo nuestro sólo le dice dónde mirar.

PALABRAS_ANCLA = 15

# Un folio suelto («280», «151»): la cabecera de página de PyMuPDF.
_RE_FOLIO = re.compile(r"^\s*\d{1,4}\s*$")
# Letras de control que el PDF de la UNAM mete en las cabeceras
# («eduardo ferrer mac-gregor\x08», «A)  \x07Los derechos…»): pdf.js no las
# pinta, así que en el ancla sobran.
_RE_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _es_titulo_corrido(linea: str) -> bool:
    """¿Es el título corrido que acompaña al folio?

    Medido sobre los 9,629 trozos: en el Diccionario va DEBAJO del folio
    («888» / «Libertades públicas»); en la Panorámica, ENCIMA («xxiv. La
    democracia y el juez constitucional» / «641») o en dos renglones («eduardo
    ferrer mac-gregor» / «panorámica del derecho procesal...» / «200»). Un
    renglón del cuerpo se distingue porque es largo, o porque PyMuPDF lo deja
    con espacio al final cuando la línea sigue, o porque termina en guion o en
    puntuación («d) El Poder Ejecutivo y el Poder Legislativo del Estado.»)."""
    crudo = _RE_CONTROL.sub("", linea or "")
    t = crudo.strip()
    if not t or len(t) > 90:
        return False
    if crudo != crudo.rstrip():          # el cuerpo sigue en el renglón de abajo
        return False
    # «panorámica del derecho procesal...»: el título corrido recortado
    # termina en puntos suspensivos, y eso no es fin de frase.
    if t.endswith("...") or t.endswith("…"):
        return True
    return not re.search(r"[-‐­.,;:]$", t)


# Lo que puede ir ENCIMA del folio (sólo la Panorámica lo hace): el capítulo
# con su número romano («VII.  Mauro Cappelletti…», «xxiv.  La democracia…»),
# el autor con su letra de control («eduardo ferrer mac-gregor\x08») o el
# título recortado («panorámica del derecho procesal...»). Nada más: en un
# trozo que empieza a media página (las páginas largas se parten en dos,
# ingestar_doctrina.py), el renglón de encima del folio es CUERPO —«aberse
# sometido a sí mismos al peso de la» / «LAS RAZONES DEL DERECHO» / «153»,
# medido en Atienza— y cortarlo movía el ancla a las notas al pie.
_RE_CAPITULO_ROMANO = re.compile(r"^\s*[IVXLCivxlc]{1,7}\.\s")


def _es_cabecera_de_encima(linea: str) -> bool:
    t = (linea or "").strip()
    if not _es_titulo_corrido(linea):
        return False
    return bool(_RE_CAPITULO_ROMANO.match(t) or _RE_CONTROL.search(t)
                or t.endswith("...") or t.endswith("…"))


def cuerpo(texto: str) -> str:
    """El texto del trozo sin la cabecera de página: el folio y el título
    corrido, si los trae arriba. Lo demás, tal cual."""
    lineas = (texto or "").splitlines()
    # Los primeros renglones con algo escrito (los vacíos no cuentan).
    idx = [i for i, l in enumerate(lineas) if l.strip()][:4]
    if not idx:
        return ""
    corte = None
    for n, i in enumerate(idx[:3]):
        if _RE_FOLIO.match(lineas[i]):
            # Todo lo de ENCIMA del folio tiene que ser cabecera; si no, ese
            # número es del cuerpo (una lista, una tabla, una nota) y no se
            # corta nada.
            if all(_es_cabecera_de_encima(lineas[j]) for j in idx[:n]):
                corte = i
            break
    if corte is None:
        return "\n".join(lineas[idx[0]:])
    resto = [i for i in idx if i > corte]
    # Debajo del folio, un renglón de título corrido (Diccionario) —sólo si
    # el folio abre la página: cuando el título corrido va ENCIMA (Panorámica),
    # lo de debajo es ya un epígrafe del texto («A)  El fallo parcialmente
    # condenatorio…») y se queda—, y sólo si detrás viene más texto: un trozo
    # de una sola línea no se vacía.
    if corte == idx[0] and len(resto) >= 2 and _es_titulo_corrido(lineas[resto[0]]):
        corte = resto[0]
    return "\n".join(lineas[corte + 1:])


def ancla(texto: str, palabras: int = PALABRAS_ANCLA) -> Optional[str]:
    """Las primeras ~15 palabras LITERALES del cuerpo del trozo.

    Literales quiere decir con sus cortes de renglón tal como los da el PDF
    («ju- dicialización», no «judicialización»): el visor normaliza igual la
    página y el ancla (minúsculas, sin acentos, sin signos), y un ancla
    «arreglada» dejaría de aparecer en la página. None si no hay texto."""
    t = _RE_CONTROL.sub("", cuerpo(texto))
    t = t.replace(" ", " ").replace(" ", " ").replace(" ", " ")
    # Los signos sueltos («—», los puntos guía de un índice «. . . .») no
    # cuentan como palabra: el visor los borra al normalizar, y quince de
    # ellos dejaban un ancla vacía.
    trozos = [w for w in t.split() if any(c.isalnum() for c in w)]
    if not trozos:
        return None
    return " ".join(trozos[:palabras])


def _entero(v: Any) -> Optional[int]:
    """int o None; lo que no sea un número entero no se inventa."""
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, str) and v.strip().isdigit():
        return int(v.strip())
    return None


def fragmento(pid: Any, pl: Dict[str, Any], score: Optional[float] = None) -> Dict[str, Any]:
    """Un punto de la colección → el fragmento con el que trabajan el chat, el
    bloque del modelo y la tarjeta. `pagina` es la que se ROTULA y se CITA:
    sólo la impresa. En el 27 % de los trozos no se conoce (25-sep-2026), y
    antes se caía a la del PDF del capítulo: la pág. 18 del capítulo 8 de
    Carbonell lleva impreso el 773, y el modelo citaba «p. 18». La del visor
    es `pagina_pdf`."""
    return {
        "id": str(pid),
        "score": score,
        "texto": pl.get("texto", "") or "",
        "autor": pl.get("autor", "") or "",
        "obra": pl.get("obra", "") or "",
        "anio": pl.get("anio"),
        "editorial": pl.get("editorial"),
        "pagina": _entero(pl.get("pagina_impresa")),
        "pagina_pdf": _entero(pl.get("pagina_pdf")),
        "pagina_impresa": _entero(pl.get("pagina_impresa")),
        "url_oficial": (pl.get("url_oficial", "") or "").split("#")[0],
        "ancla": ancla(pl.get("texto", "") or ""),
    }


def origen_de(f: Dict[str, Any]) -> str:
    """«Autor, «Obra», año»: como se rotula la fuente en la app."""
    anio = f", {f['anio']}" if f.get("anio") else ""
    return f"{f.get('autor') or ''}, «{f.get('obra') or ''}»{anio}"


def contrato(f: Dict[str, Any], score: float) -> Dict[str, Any]:
    """Un fragmento → los campos de SearchResult con el contrato del visor."""
    url = (f.get("url_oficial") or "").split("#")[0] or None
    return dict(
        id=str(f["id"]),
        score=score,
        texto=f.get("texto") or "",
        ref=f"p. {f['pagina']}" if f.get("pagina") else "",
        origen=origen_de(f),
        jurisdiccion="Doctrina",
        silo=COLECCION,
        # El enlace apunta a la obra EN SU REPOSITORIO. Nunca a nuestro
        # texto. Sin «#page»: la página viaja aparte, en `pagina`.
        pdf_url=url,
        url_oficial=url,
        pagina=_entero(f.get("pagina_pdf")),
        pagina_impresa=_entero(f.get("pagina_impresa")),
        ancla=f.get("ancla"),
        obra=f.get("obra") or None,
        autor=f.get("autor") or None,
        anio=_entero(f.get("anio")),
    )


def bloque_para_prompt(frags: List[Dict[str, Any]]) -> str:
    """Lo que ve el MODELO. Con la regla de uso pegada al material."""
    if not frags:
        return ""
    lineas = [
        "<doctrina>",
        "Fragmentos de obras jurídicas de referencia, con su cita. Reglas:",
        "1. ÚSALOS para enriquecer el concepto, atribuyendo SIEMPRE autor y",
        "   obra, y la página SÓLO si aquí viene, p. ej. (Atienza, Las razones",
        "   del derecho, p. 45). Si no viene página, no la pongas ni la inventes.",
        "2. Si citas textual, MÁXIMO 40 palabras y entre comillas — es derecho",
        "   de cita, no reproducción. Sólo puedes citar textual lo que esté",
        "   AQUÍ; jamás de memoria.",
        "3. La doctrina ILUSTRA; el fundamento son la ley y la jurisprudencia.",
        "",
    ]
    for f in frags:
        lineas.append(
            f"— {f['autor']}, «{f['obra']}»" + (f", p. {f['pagina']}" if f.get("pagina") else "")
            + f":\n{f['texto'][:1200]}\n")
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
    los fragmentos.

    LA ATRIBUCIÓN EXIGE MÁS QUE UN APELLIDO (7-ago-2026). Ferrer Mac-Gregor es
    autor del Diccionario Y ministro de la SCJN: con el apellido como única
    señal, una comilla tomada de una tesis o de un voto suyo se marcaba como
    «cita doctrinal sin verificar» y la tarjeta advertía en falso. Ahora la
    comilla es doctrinal sólo si en su ventana aparece el [Doc ID] de un
    fragmento doctrinal, o el apellido JUNTO a una seña de la obra (su título
    o el patrón «p. N» de página) — una tesis se cita por registro, no por
    página de libro."""
    if not frags:
        return 0
    corpus = normalizar(" ".join(f["texto"] for f in frags))
    apellidos = set()
    tokens_obra = set()
    ids = {str(f.get("id") or "") for f in frags if f.get("id")}
    for f in frags:
        for token in re.split(r"[,y\s]+", str(f["autor"])):
            if len(token) > 3:
                apellidos.add(normalizar(token))
        for token in re.split(r"[\s,«»]+", str(f["obra"])):
            if len(token) > 5:
                tokens_obra.add(normalizar(token))
    fallidas = 0
    for m in _RE_COMILLAS.finditer(respuesta or ""):
        cita = m.group(1)
        ventana_cruda = respuesta[max(0, m.start() - 260):m.end() + 260]
        ventana = normalizar(ventana_cruda)
        con_doc_id = any(i and i in ventana_cruda for i in ids)
        con_autor = any(a in ventana for a in apellidos)
        con_obra = any(o in ventana for o in tokens_obra) or re.search(r"\bp\.?\s*\d{1,4}\b", ventana_cruda)
        if not (con_doc_id or (con_autor and con_obra)):
            continue          # no se atribuye a la doctrina: no es nuestro caso
        if normalizar(cita) not in corpus:
            fallidas += 1
    return fallidas
