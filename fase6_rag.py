"""El RAG del estudio de fondo: de un problema jurídico al material para fundarlo.

POR QUÉ ESTE MÓDULO BUSCA DISTINTO QUE EL CHAT
══════════════════════════════════════════════

La medición sobre 404 tesis realmente citadas en 828 engroses del disco KINGSTON
dejó dos conclusiones que aquí se aprovechan enteras:

    v3 conceptual → vector RUBRO ..... 50% en 1ª posición · 75% en top-10  ← ésta
    v3 conceptual → vector texto ..... 44% / 69%
    v3 prosa      → vector RUBRO .....  3% / 15%          ← lo peor de todo

Es decir: **cómo se pregunta importa diez veces más que con qué se busca**, y el
vector del rubro sólo rinde cuando la consulta es una pregunta conceptual.

Y aquí se da la coincidencia que hace barato todo esto: la Fase 3 ya redacta los
problemas jurídicos **como preguntas** —«¿Puede condenarse a pensión
compensatoria invirtiendo la carga de la prueba?»— porque así es como se
resuelven en una sentencia. O sea que la entrada del RAG ya viene en el formato
que mejor recupera, sin reformular nada ni pagar una llamada extra al modelo.

LA OBLIGATORIEDAD MANDA EN EL ORDEN
═══════════════════════════════════
`vincula` separa 17.930 tesis obligatorias de 53.998 orientadoras. En un chat da
igual el orden; en una sentencia no: una jurisprudencia obligatoria en contra
cambia el sentido, y una aislada sólo ilustra. Se buscan las dos, pero las
obligatorias van primero y se marcan como tales para que el estudio las trate
como lo que son.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import logging
import re
from typing import Awaitable, Callable, Optional

log = logging.getLogger("fase6_rag")

import fase6_estudio as f6

COLECCION_JURIS = "jurisprudencia_nacional_v3"
VECTOR_RUBRO = "rubro"          # el ganador medido, con pregunta conceptual
COLECCION_FEDERAL = "leyes_federales"

TESIS_POR_PROBLEMA = 6          # 6 entran en el prompt sin ahogar el material
NORMAS_POR_PROBLEMA = 4


# El rubro de una tesis local lo dice sin ambigüedad: «(LEGISLACIÓN DEL ESTADO
# DE PUEBLA)». Un criterio sobre el código de otra entidad no rige aquí, por
# obligatorio que sea en su circuito.
_RX_LEGISLACION = re.compile(r"LEGISLACI[ÓO]N\s+(?:DEL?\s+)?(?:ESTADO\s+DE\s+)?"
                             r"([A-ZÁÉÍÓÚÑ ]{4,40}?)\s*\)", re.I)

_ESTADO_DE_COLECCION = {
    "leyes_queretaro": "QUERETARO", "leyes_jalisco": "JALISCO",
    "leyes_cdmx": "CIUDAD DE MEXICO", "leyes_nuevo_leon": "NUEVO LEON",
    "leyes_guanajuato": "GUANAJUATO", "leyes_puebla": "PUEBLA",
}


def _sin_acentos(x: str) -> str:
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFKD", (x or "").upper())
                   if not unicodedata.combining(c))


def _de_otro_estado(t: dict, coleccion: Optional[str]) -> bool:
    """True si la tesis interpreta la legislación de OTRA entidad."""
    m = _RX_LEGISLACION.search(t.get("rubro", "") or "")
    if not m:
        return False
    mia = _ESTADO_DE_COLECCION.get((coleccion or "").lower(), "")
    suya = _sin_acentos(m.group(1)).strip()
    return bool(mia) and mia not in suya and suya not in mia


# ═══════════════════════════════════════════════════════════════════════════
# LA JERARQUÍA: PRIMERO SI VINCULA, DESPUÉS QUIÉN LO DIJO
# ═══════════════════════════════════════════════════════════════════════════
# David: «no olvides que la jerarquía del orden pese».
#
# El orden era `(no es SCJN, no es obligatoria, …)`, y con eso una TESIS
# AISLADA de la Primera Sala le ganaba a una JURISPRUDENCIA de tribunal
# colegiado. Comprobado ejecutándolo. Está al revés: la aislada NO vincula a
# nadie —orienta— y la jurisprudencia del colegiado SÍ obliga en su circuito.
# El estudio que se apoya en la primera se apoya en algo que la contraparte
# puede discutir; el que se apoya en la segunda, no.
#
# El orden correcto es el de la Ley de Amparo, no el del prestigio del emisor:
#   1º  si VINCULA
#   2º  quién lo dijo, dentro de cada grupo
#   3º  cuántas veces lo cita el circuito para esta cuestión
#   4º  si la tesis es de legislación de otra entidad
# CORREGIDO POR DAVID, y la corrección importa. Yo había puesto lo VINCULANTE
# primero y la instancia después, con el argumento de que una aislada no obliga
# y una jurisprudencia de colegiado sí. Él:
#
#   «Las de la SCJN son preferentes aunque sean aisladas. En segundo lugar las
#    de colegiados porque esas son optativas para colegiados de otros
#    circuitos. Sólo las de plenos regionales irían en segundo lugar.»
#
# Tiene razón y mi razonamiento estaba incompleto: la jurisprudencia de un
# colegiado obliga en SU circuito, pero para cualquier otro es optativa —y el
# acervo es nacional—, mientras que un criterio de la Corte, aunque sea
# aislado, orienta a todos y nadie lo discute. En un tribunal colegiado se cita
# antes una aislada de la Primera Sala que una jurisprudencia de un colegiado
# de otro circuito.
#
# El orden queda: Corte · Plenos Regionales · Colegiados · lo demás; y dentro
# de cada grupo, primero lo que vincula.
_RANGO = (("PLENO REGIONAL", 1), ("PLENOS REGIONALES", 1), ("PLENO", 0),
          ("SUPREMA CORTE", 0), ("PRIMERA SALA", 0), ("SEGUNDA SALA", 0),
          ("SALA", 0), ("TRIBUNALES COLEGIADOS", 2), ("COLEGIADO", 2))


def _rango_instancia(t: dict) -> int:
    """0 Suprema Corte · 1 Plenos Regionales · 2 Colegiados · 3 lo demás."""
    inst = _sin_acentos(t.get("instancia", ""))
    for clave, r in _RANGO:
        if clave in inst:
            return r
    return 3


def _es_scjn(t: dict) -> bool:
    """Pleno o Salas. David: «preferentemente jurisprudencia de la Suprema Corte»."""
    inst = _sin_acentos(t.get("instancia", ""))
    return any(x in inst for x in ("PRIMERA SALA", "SEGUNDA SALA", "PLENO",
                                   "SUPREMA CORTE"))


def _tesis_de(p: dict) -> dict:
    return {
        "registro": str(p.get("registro") or ""),
        "instancia": p.get("instancia") or "",
        "rubro": p.get("rubro") or "",
        "texto": p.get("texto") or "",
        "tipo": p.get("tipo") or "",
        "obligatoria": bool(p.get("vincula")),
        "localizacion": p.get("localizacion") or "",
    }


def _norma_de(p: dict) -> dict:
    return {
        # CUATRO NOMBRES PARA EL MISMO DATO, y cada colección usa el suyo:
        # `cuerpo_legal_oficial` en leyes_federales, `origen` en el bloque
        # constitucional, `ley` y `cuerpo_legal` en el silo laboral. Leer sólo
        # uno dejaba el nombre VACÍO en las 22 normas del silo, y como el
        # compositor ya no transcribe un artículo cuya ley no puede identificar
        # —esa regla se puso hoy, para que no vuelva a pegar el texto de otra
        # ley—, el documento se habría quedado sin una sola transcripción.
        "cuerpo_legal": (p.get("cuerpo_legal_oficial") or p.get("cuerpo_legal")
                         or p.get("ley") or p.get("origen") or p.get("ref") or ""),
        "articulo": p.get("articulo_num") or "",
        "texto": p.get("texto") or p.get("contenido") or "",
        "entidad": p.get("entidad") or "",
    }


async def _buscar(qdrant, coleccion: str, vector: str, v: list[float],
                  limite: int, filtro=None) -> list[dict]:
    """Una consulta a Qdrant, tolerante con la colección que falta y RUIDOSA con
    el resto.

    La primera versión devolvía `[]` ante cualquier excepción y el estudio salía
    sin una sola tesis, indistinguible de «no hay nada que citar». El error real
    era que el cliente ya no tiene `.search` —hoy es `query_points`— y quedó
    escondido tras el `except`. Una búsqueda que falla se REGISTRA; sólo el
    estado sin ingestar es un vacío legítimo.
    """
    try:
        r = qdrant.query_points(collection_name=coleccion, query=v,
                                using=vector, limit=limite,
                                query_filter=filtro, with_payload=True)
        if inspect.isawaitable(r):          # AsyncQdrantClient, el de main.py
            r = await r
        return [p.payload or {} for p in r.points]
    except Exception as e:
        texto = str(e).lower()
        if "not found" in texto or "doesn" in texto:
            log.info("colección %s sin ingestar; se sigue sin ella", coleccion)
        else:
            log.error("búsqueda fallida en %s/%s: %s: %s",
                      coleccion, vector, type(e).__name__, e)
        return []


# ═══════════════════════════════════════════════════════════════════════════
# EL ARTÍCULO ENTERO, NO EL TROZO QUE CASÓ
# ═══════════════════════════════════════════════════════════════════════════
# En el ADL 382/2024 —un trabajador despedido por faltas— el motor propuso
# INFUNDADO cinco veces razonando sobre si las incapacidades se habían entregado
# a tiempo. Nunca discutió lo único que decide el asunto: que el artículo 47,
# fracción X, de la Ley Federal del Trabajo exige que las faltas sean «SIN CAUSA
# JUSTIFICADA», y una incapacidad médica real es causa justificada, se entregue
# el papel cuando se entregue.
#
# No fue un fallo de razonamiento. ES QUE NUNCA LO LEYÓ. El artículo 47 sí
# llegó al material… con 600 caracteres: el encabezado y la fracción I. El
# artículo tiene quince fracciones y está troceado; la búsqueda por parecido
# devuelve EL TROZO que casó, y casó el primero. La fracción X se quedó fuera.
#
# Un trozo no es un artículo. Cuando la ley se trae para fundar, se trae entera:
# `fase_normas` ya lo hacía para las notas al pie —«el artículo entero: viene
# troceado y un trozo no es el artículo»— y el material que sostiene el
# RAZONAMIENTO iba sin ese cuidado, que es donde más falta hace.

async def _completar(qdrant, coleccion: str, norma: dict) -> dict:
    """Los demás trozos del mismo artículo y la misma ley, en orden."""
    from qdrant_client.models import FieldCondition, Filter, MatchValue
    num, ley = norma.get("articulo"), str(norma.get("cuerpo_legal") or "")
    if not num or not ley:
        return norma
    # EL SILO YA GUARDA EL ARTÍCULO ENTERO —un punto por artículo, que es su
    # primera regla de diseño—, así que aquí no hay nada que completar y el
    # scroll sería un viaje en balde por cada norma.
    if coleccion in SILO_POR_MATERIA.values():
        return norma
    try:
        r = qdrant.scroll(
            collection_name=coleccion,
            scroll_filter=Filter(must=[FieldCondition(
                key="articulo_num", match=MatchValue(value=int(num)))]),
            limit=60, with_payload=True)
        if inspect.isawaitable(r):
            r = await r
        pts = r[0] if isinstance(r, tuple) else r
    except Exception as e:
        log.error("no se pudo completar el artículo %s: %s", num, e)
        return norma
    # SÓLO LOS DE LA MISMA LEY. El artículo 47 existe en decenas de códigos y
    # juntarlos daría un texto que no es de ninguno.
    suyos = [x.payload for x in (pts or [])
             if str((x.payload or {}).get("cuerpo_legal_oficial")
                    or (x.payload or {}).get("ley") or "") == ley]
    if len(suyos) < 2:
        return norma
    partes, visto = [], set()
    for pl in sorted(suyos, key=lambda z: int(z.get("chunk_index") or 0)):
        # La migaja de cabecera «[Ley … | CAPITULO IV …]» se repite en cada
        # trozo; una vez basta y en los demás estorba.
        txt = re.sub(r"^\s*\[[^\]]{0,250}\]\s*", "",
                     " ".join(str(pl.get("texto") or "").split()))
        if txt and txt not in visto:
            visto.add(txt)
            partes.append(txt)
    entero = " ".join(partes)
    if len(entero) > len(str(norma.get("texto") or "")):
        norma = dict(norma)
        norma["texto"] = entero[:6000]
        norma["completo"] = True
    return norma


# ═══════════════════════════════════════════════════════════════════════════
# EL SILO POR MATERIA
# ═══════════════════════════════════════════════════════════════════════════
# David, 31-ago-2026: «si apuntamos el RAG a las normas que pudieran resultar
# aplicables por materia, la recuperación sería significativamente superior».
#
# Lo medí antes de creérmelo, con las consultas reales del ADL 382/2024:
#
#   «¿La Junta valoró indebidamente las incapacidades médicas…?»
#     corpus general → Ley del ISSSTE 58, 66, 37 · Responsabilidades Admvas 107
#     silo laboral   → LFT 481, 492, 493, 497, 491
#
# El artículo 37 de la Ley del ISSSTE —licencias para padres de menores con
# cáncer— es exactamente el que acabó transcrito en un despido de un enfermero
# del IMSS. Con el silo deja de ser alcanzable.
#
# LO QUE EL SILO NO ARREGLA, y hay que decirlo: en otra consulta el general
# devolvió cinco artículos de Responsabilidades Administrativas y el silo cinco
# de la Ley Orgánica del PJF, cuando la respuesta correcta es el artículo 784 de
# la LFT. Filtrar por materia elimina la ley AJENA; la precisión dentro del
# corpus correcto es otro problema y sigue abierto.
# LA TABLA DE SILOS. Cada materia apunta al suyo, y su composición se midió
# sobre 1,600 trozos de sentencias reales de esa materia —no se eligió a ojo—.
# El núcleo es común a las cuatro (Constitución, Ley de Amparo, Convención
# Americana, Pacto Internacional, Ley Orgánica del PJF) porque todo amparo lo
# lleva; lo demás lo puso la medición.
#
# LA LEY DEL ESTADO NO ESTÁ AQUÍ, y es deliberado. En civil los tres códigos
# civiles estatales más citados son de Jalisco, la Ciudad de México y Querétaro:
# no hay uno que sirva para todos. Se apunta aparte, por el ACTO RECLAMADO y no
# por el circuito, porque el Vigésimo Segundo revisa actos de Querétaro y de
# Hidalgo y el acto dice cuál rige.
SILO_POR_MATERIA = {
    "laboral": "leyes_laboral",              # 3,205 art · 9 ordenamientos
    "civil": "leyes_civil",                  # 5,851 art · 9
    "administrativa": "leyes_administrativa",  # 3,092 art · 12
    "penal": "leyes_penal",                  # 2,187 art · 8
}

# ═══════════════════════════════════════════════════════════════════════════
# SEMBRAR CON LO QUE CITARON LOS TRIBUNALES
# ═══════════════════════════════════════════════════════════════════════════
# La versión barata de la idea de David —«conecta el corpus iuris con los
# holdings»— antes de construir ningún grafo. En vez de buscar la norma por
# parecido con la pregunta, se mira QUÉ ARTÍCULOS citaron los colegiados cuando
# resolvieron el mismo tema, y se traen ésos por número.
#
# MEDIDO, y es contundente:
#
#   «¿A quién corresponde la carga de probar la causa de rescisión?»
#     semántico → Ley Orgánica del PJF 179, 177, 191…   (el 784 no aparece)
#     sembrado  → 47, 784, 517, 48…                     ✓
#
#   «¿Debe suplirse la deficiencia de la queja?»
#     semántico → LFT 893, 862, 872…                    (el 79 no aparece)
#     sembrado  → 76, 107, 79, 123…                     ✓
#
# Donde el parecido semántico falla del todo, el sembrado acierta. La razón es
# obvia una vez vista: la pregunta «¿a quién corresponde la carga de la prueba?»
# no se parece al TEXTO del artículo 784 —que es una lista de hechos—, pero
# treinta y cuatro tribunales lo citaron al resolver eso mismo. El parecido
# busca palabras; el sembrado usa lo que ya se decidió.
#
# NO SUSTITUYE AL SEMÁNTICO: lo precede. El sembrado trae lo que se usa siempre
# y se quedaría corto ante una cuestión nueva; el semántico cubre esa cola.

# «artículo 784 de la Ley Federal del Trabajo», con la ley al final de la cita.
_RX_ART_CITADO = re.compile(
    r"art[íi]culos?\s+(\d{1,4})(?:\s*,?\s*fracci[óo]n(?:es)?\s+[IVXLC]+)?[^.;]{0,90}?"
    r"(Ley\s+Federal\s+del\s+Trabajo|Ley\s+de\s+Amparo|Constituci[óo]n|"
    r"Ley\s+del\s+Seguro\s+Social|C[óo]digo\s+Federal\s+de\s+Procedimientos|"
    r"Ley\s+Federal\s+de\s+los\s+Trabajadores)", re.I)
_LEY_CANONICA = {
    "ley federal del trabajo": "Ley Federal del Trabajo",
    "ley de amparo": "Ley de Amparo",
    "constitucion": "Constitución Política de los Estados Unidos Mexicanos",
    "ley del seguro social": "Ley del Seguro Social",
    "codigo federal de procedimientos": "Código Federal de Procedimientos Civiles",
    "ley federal de los trabajadores": "Ley Federal de los Trabajadores al Servicio del Estado",
}
CIRCUITOS_CON_ESTUDIO = ("1", "2", "3", "4", "22")
HOLDINGS_A_SEMBRAR = 10      # cuántas sentencias se leen
ARTICULOS_SEMBRADOS = 6      # cuántos artículos entran por esta vía


def _canonica(nombre: str) -> str:
    import unicodedata
    x = unicodedata.normalize("NFKD", (nombre or "").lower())
    x = "".join(c for c in x if not unicodedata.combining(c)).strip()
    return _LEY_CANONICA.get(x, "")


async def _sembrar(qdrant, silo: str, vector, materia: str) -> list:
    """Los artículos que los colegiados citaron al resolver este mismo tema."""
    from collections import Counter
    from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue
    grafias = {"laboral": ["laboral", "Trabajo"], "civil": ["civil", "Civil"],
               "administrativa": ["administrativa", "Administrativa"],
               "penal": ["penal", "Penal"]}.get((materia or "").lower())
    if not (silo and grafias):
        return []
    try:
        r = qdrant.query_points(
            collection_name="sentencias_holdings", query=vector, using="dense",
            query_filter=Filter(must=[FieldCondition(key="materia",
                                                     match=MatchAny(any=grafias))]),
            limit=HOLDINGS_A_SEMBRAR, with_payload=["circuito"])
        if inspect.isawaitable(r):
            r = await r
        hold = [(str(p.id), str((p.payload or {}).get("circuito") or ""))
                for p in (getattr(r, "points", None) or [])]
    except Exception as e:
        log.error("sembrado: no se pudo sondear holdings: %s", e)
        return []

    async def _estudio(hid, circ):
        if circ not in CIRCUITOS_CON_ESTUDIO:
            return ""
        try:
            rr = qdrant.scroll(
                collection_name=f"sentencias_ef_c{circ}",
                scroll_filter=Filter(must=[FieldCondition(
                    key="holding_id", match=MatchValue(value=hid))]),
                limit=60, with_payload=["chunk_text"])
            if inspect.isawaitable(rr):
                rr = await rr
            pts = rr[0] if isinstance(rr, tuple) else rr
            return " ".join(str((p.payload or {}).get("chunk_text") or "") for p in pts)
        except Exception:
            return ""

    textos = await asyncio.gather(*[_estudio(h, c) for h, c in hold])
    cuenta = Counter()
    for txt in textos:
        for m in _RX_ART_CITADO.finditer(txt):
            ley = _canonica(m.group(2))
            if ley:
                cuenta[(ley, int(m.group(1)))] += 1
    if not cuenta:
        return []

    from qdrant_client.models import FieldCondition as FC, Filter as F, MatchValue as MV

    async def _traer(ley, num):
        try:
            rr = qdrant.scroll(collection_name=silo, limit=4, with_payload=True,
                               scroll_filter=F(must=[
                                   FC(key="articulo_num", match=MV(value=num)),
                                   FC(key="ley", match=MV(value=ley))]))
            if inspect.isawaitable(rr):
                rr = await rr
            pts = rr[0] if isinstance(rr, tuple) else rr
            return [p.payload for p in (pts or [])][:1]
        except Exception:
            return []

    top = [k for k, _ in cuenta.most_common(ARTICULOS_SEMBRADOS)]
    traidos = await asyncio.gather(*[_traer(l, n) for l, n in top])
    fuera = []
    for (ley, num), pl in zip(top, traidos):
        if pl:
            n = _norma_de(pl[0])
            n["sembrada"] = cuenta[(ley, num)]
            fuera.append(n)
    if fuera:
        log.info("sembrado: %d artículos de lo que citaron los colegiados",
                 len(fuera))
    return fuera


MODELO_CONSULTA = os.getenv("MODELO_CONSULTA", "gpt-5.6-luna")

_PROMPT_CONCEPTO = """Eres secretario de un Tribunal Colegiado. Traduce el problema
jurídico al lenguaje de los RUBROS del Semanario Judicial, que es como se indexa
la jurisprudencia mexicana: sintagmas nominales en versales, la FIGURA JURÍDICA
primero y sus notas después, sin verbos conjugados ni pronombres del caso.

NO describas los hechos ni nombres a las partes. Nombra la INSTITUCIÓN que
gobierna el punto.

  problema:  ¿La responsable valoró indebidamente el dictamen pericial?
  rubro:     PRUEBA PERICIAL. VALORACIÓN. FACULTADES DEL JUZGADOR

Devuelve SÓLO un JSON, sin texto alrededor:
{{"precisa": "<el rubro de la figura EXACTA del problema>",
  "amplia":  "<el rubro del GÉNERO al que esa figura pertenece, para alcanzar
              por analogía cuando no exista criterio sobre el caso concreto>"}}

MATERIA: {materia}
PROBLEMA: {problema}"""

_RX_JSON_C = re.compile(r"\{.*\}", re.S)


async def consulta_conceptual(cliente, problema: str, materia: str = "") -> dict:
    """El problema, dicho como lo diría un rubro. Es LA palanca de la búsqueda.

    POR QUÉ EXISTE. La colección v3 y el vector `rubro` son la configuración
    ganadora medida —50% en primera posición frente al 35% de la v2— PERO sólo
    con pregunta conceptual: con prosa, ese mismo vector es lo PEOR de todo lo
    probado. Este fichero lo decía en su cabecera desde el principio y el
    código mandaba la pregunta entera, con sus signos de interrogación.

    Medido sobre los tres problemas de la revisión 410/2026, puntuación media
    del top-10 contra el vector `rubro`:

        pregunta tal cual .... 0.612 · 0.596 · 0.623
        rubro «precisa» ...... 0.729 · 0.776 · 0.736
        rubro «amplia» ....... 0.674 · 0.725 · 0.820

    Y no es sólo puntuación: con la pregunta cruda, el problema del
    emplazamiento devolvía seis tesis sobre el JUICIO DE NULIDAD y ninguna
    sobre TERCERO EXTRAÑO POR EQUIPARACIÓN, que es la figura que lo decide. El
    problema de la suplencia devolvía «INCONFORMIDAD. LA SUPREMA CORTE DEBE
    SUPLIR LA QUEJA DEFICIENTE…», que David tachó del proyecto porque no
    aplicaba: no fue capricho del modelo, se la dio la búsqueda.

    QUITAR EL ANDAMIO NO BASTA —medido: 0.604 frente a 0.612, y cero
    coincidencias con lo que trae la conceptual—. No es una resta: es una
    traducción, y hay que formularla.

    LAS DOS CONSULTAS, y la segunda es la que pidió David: «si no la hay,
    existe una tesis que resuelve por analogía». La «amplia» busca el GÉNERO
    de la figura, y en el problema 3 puntuó más alto que la precisa (0.820) y
    trajo obligatorias distintas.
    """
    if cliente is None or not (problema or "").strip():
        return {}
    kw = dict(model=MODELO_CONSULTA, max_completion_tokens=4000,
              reasoning_effort="low",
              messages=[{"role": "user", "content": _PROMPT_CONCEPTO.format(
                  materia=materia or "no consta", problema=problema)}])
    try:
        import llamada_modelo as _lm
        r = await _lm.crear(cliente, **kw)
        m = _RX_JSON_C.search((r.choices[0].message.content or "").strip())
        d = json.loads(m.group(0)) if m else {}
        return {k: str(d.get(k, ""))[:220] for k in ("precisa", "amplia")
                if str(d.get(k, "")).strip()}
    except Exception as e:
        # SE SIGUE CON LA PREGUNTA CRUDA. Peor búsqueda, pero búsqueda: que se
        # caiga el traductor no puede dejar al secretario sin acervo.
        print(f"   ⚠️ RAG: no se pudo formular la consulta conceptual: "
              f"{type(e).__name__}")
        return {}


async def material_para(qdrant, embed_juris, embed_leyes,
                        problema: str, coleccion_estatal: Optional[str] = None,
                        materia: str = "", cliente=None,
                        contexto: str = "") -> f6.Material:
    """El material verificado para UN problema jurídico.

    `embed_juris` vectoriza con el modelo de la v3 (3072 dim) y `embed_leyes`
    con el de las colecciones de leyes: son modelos distintos y cruzarlos
    devuelve ruido con buena puntuación, que es la peor clase de error.
    """
    # LAS TRES ANCLAS. La pregunta cruda se conserva como red —si el traductor
    # falla, la búsqueda sigue siendo la de siempre— y sobre ella se suman las
    # dos conceptuales, que son las que de verdad alcanzan el rubro.
    #
    # Es el mismo patrón que ya estaba medido en el chat: «con una sola ancla,
    # 2 de 3; con las dos, 5 de 5».
    _c = await consulta_conceptual(cliente, problema, materia)
    anclas = [x for x in (_c.get("precisa"), _c.get("amplia"), problema) if x]
    # ── LO QUE SABE EL SECRETARIO, COMO ANCLA PROPIA ──────────────────────
    #
    # David: «primero debe presentarse todo el contexto jurídico y después
    # buscar la solución… así el sistema va a tener mejor capacidad de buscar
    # jurisprudencia o las normas aplicables al caso».
    #
    # Va como ANCLA APARTE, no pegado a la pregunta, y ésa es la diferencia que
    # importa. Está medido en este mismo sistema que el andamio interrogativo
    # pesa más en el vector que el concepto jurídico: la consulta literal del
    # derecho del tanto devolvía Ley del Deporte y Código Fiscal, y el concepto
    # descrito devolvía los artículos exactos. Mezclar el contexto DENTRO de la
    # pregunta diluiría las tres anclas que hoy funcionan; sumarlo como cuarta
    # sólo puede añadir.
    #
    # Se recorta a 600 caracteres porque lo que se vectoriza es un concepto, no
    # un expediente: un texto largo se promedia hasta no significar nada.
    _ctx = " ".join((contexto or "").split())[:600]
    if len(_ctx) >= 40:
        anclas.append(_ctx)
    _vs = await asyncio.gather(*[embed_juris(a) for a in anclas],
                               embed_leyes(problema))
    v_leyes = _vs[-1]
    v_juris = _vs[0]

    # EL SILO SUSTITUYE, NO SE SUMA: si se buscara también en el corpus general
    # volvería a entrar la ley ajena por la puerta de atrás, que es justo lo que
    # el silo existe para cerrar.
    silo = SILO_POR_MATERIA.get((materia or "").strip().lower())
    colecciones = [silo] if silo else [c for c in (coleccion_estatal, COLECCION_FEDERAL) if c]
    tareas = [_buscar(qdrant, COLECCION_JURIS, VECTOR_RUBRO, v,
                      TESIS_POR_PROBLEMA * 2) for v in _vs[:-1]]
    tareas += [_buscar(qdrant, c, "dense", v_leyes, NORMAS_POR_PROBLEMA)
               for c in colecciones]
    # (res[:n_anclas] son las tesis; res[n_anclas:] son las normas)
    # LA CO-CITACIÓN VA EN PARALELO con las demás búsquedas: no alarga nada.
    res, _coc = await asyncio.gather(
        asyncio.gather(*tareas),
        tesis_cocitadas(qdrant, embed_leyes, problema))

    # EL ORDEN, CORREGIDO. La primera versión de esto penalizaba la tesis por
    # venir de otra entidad y la mandaba al fondo. Estaba mal, y el barrido de
    # 139 documentos de este tribunal lo demuestra: hay decenas de criterios
    # sobre legislación ajena invocados con toda naturalidad —la XVI.1o.A. J/54,
    # sobre la Ley de Hacienda de GUANAJUATO, sostiene el interés jurídico en
    # SEIS amparos queretanos— y CERO aplicaciones de ley de otra entidad.
    #
    # Lo que no entra es la LEY ajena, y eso no se arregla ordenando tesis: se
    # arregla en el prompt y en el verificador. Aquí sólo se ordena por peso.
    #
    #   1. Que venga de la Suprema Corte (Pleno o Salas): es obligatoria por el
    #      artículo 217 y la legislación que interpretó es irrelevante.
    #   2. Que sea obligatoria.
    #   3. A igualdad de lo anterior, antes la de Querétaro que la de fuera.
    # LAS TRES LISTAS SE FUNDEN, sin repetir registro. El orden de mérito lo
    # pone el criterio de abajo, no el ancla por la que entró.
    _crudo, _vistos = [], set()
    for lista in res[:len(_vs) - 1]:
        for p in lista:
            t = _tesis_de(p)
            if t["registro"] and t["registro"] not in _vistos:
                _vistos.add(t["registro"])
                _crudo.append(t)
    # LOS CO-CITADOS ENTRAN EN LA MISMA LISTA, no detrás. Añadirlos al final
    # es regalárselos al recorte del prompt: ya pasó con las tesis de la
    # técnica y no llegó ninguna.
    for _t in _coc:
        if _t["registro"] and _t["registro"] not in _vistos:
            _vistos.add(_t["registro"])
            _crudo.append(_t)
        else:
            # Ya la traía la semántica: se le anota que ADEMÁS la cita el
            # circuito, que es lo que le da peso en el orden.
            for _y in _crudo:
                if _y["registro"] == _t["registro"]:
                    _y["veces"] = _t.get("veces", 0)
                    _y["cocitada"] = True
                    break
    tesis = _crudo
    # EL ORDEN, con la co-citación dentro: entre dos criterios que pesan igual,
    # manda el que el circuito usa de verdad para esta cuestión.
    tesis.sort(key=lambda t: (_rango_instancia(t),
                              not t["obligatoria"],
                              -int(t.get("veces") or 0),
                              _de_otro_estado(t, coleccion_estatal)))
    vistos: set[str] = set()
    unicas: list[dict] = []
    for t in tesis:
        if t["registro"] and t["registro"] not in vistos:
            vistos.add(t["registro"])
            unicas.append(t)

    # CADA NORMA CON LA COLECCIÓN DE LA QUE SALIÓ, para poder completarla
    # después sin volver a construir la lista. Reconstruirla era el fallo: lo
    # sembrado se añadía aquí y `_completar` lo borraba tres líneas más abajo al
    # rehacer `normas` desde `res`. Dos arreglos míos del mismo día, cada uno
    # correcto por su cuenta, peleándose.
    pares = []
    # LAS NORMAS EMPIEZAN DONDE ACABAN LAS ANCLAS. Esto decía `res[1:]`, que
    # era correcto con UNA búsqueda de tesis; con tres anclas, `res[1]` y
    # `res[2]` son tesis y se colaban como si fueran leyes.
    for grupo, col in zip(res[len(_vs) - 1:], colecciones):
        pares += [(col, _norma_de(p)) for p in grupo]

    # LO SEMBRADO VA DELANTE. Son los artículos que los tribunales usaron de
    # verdad para esta cuestión; lo semántico cubre la cola.
    if silo:
        sembradas = await _sembrar(qdrant, silo, v_leyes, materia)
        vistos = {(str(n.get("cuerpo_legal")), str(n.get("articulo"))) for n in sembradas}
        pares = [(silo, n) for n in sembradas] + [
            (c, n) for c, n in pares
            if (str(n.get("cuerpo_legal")), str(n.get("articulo"))) not in vistos]

    # Cada norma, completada con el resto de su articulado. Van en paralelo y
    # es un scroll por artículo: el mismo precio que ya paga la nota al pie.
    normas = list(await asyncio.gather(
        *[_completar(qdrant, c, n) for c, n in pares])) if pares else []

    return f6.Material(tesis=unicas[:TESIS_POR_PROBLEMA],
                       normas=normas[:NORMAS_POR_PROBLEMA * 2],
                         principios=list(getattr(tesis_cocitadas, 'ultimos_principios', []) or []))


async def material_del_caso(qdrant, embed_juris, embed_leyes,
                            problemas: list[str],
                            coleccion_estatal: Optional[str] = None,
                            materia: str = "", cliente=None,
                            contexto: str = "") -> f6.Material:
    """Un solo Material con lo de TODOS los problemas, sin repetir tesis.

    El estudio se escribe de una vez —es una sola pieza de prosa— así que el
    material también se le entrega de una vez, deduplicado por registro y por
    artículo. Si se le mandara por problema, citaría la misma tesis tres veces.
    """
    # Se acepta la pregunta suelta o el problema entero de la Fase 3: que el
    # módulo aguante las dos formas cuesta tres líneas y evita repetir el fallo
    # en cada sitio que lo llame.
    preguntas = []
    for p in (problemas or []):
        q = p.get("pregunta", "") if isinstance(p, dict) else str(p or "")
        if q.strip():
            preguntas.append(q.strip())

    partes = await asyncio.gather(*[
        material_para(qdrant, embed_juris, embed_leyes, p, coleccion_estatal,
                      materia, cliente, contexto)
        for p in preguntas])

    tesis, normas = [], []
    r_vistos, n_vistos = set(), set()
    # DE QUÉ PROBLEMA VIENE CADA TESIS. La búsqueda ya es por problema, pero al
    # aplanar se perdía esa procedencia y la fase que propone recibía ocho
    # tesis sueltas para tres o siete problemas. Medido sobre cinco asuntos
    # reales: 191 tesis recuperadas y 10 invocadas, el 5,2 %; en el ADA 47/2025,
    # seis de siete problemas se resolvieron SIN un solo criterio.
    #
    # Sigue sin repetirse una tesis —el estudio es una sola pieza de prosa y
    # citarla tres veces la abarata—: lo que se guarda es a qué problemas
    # sirve, para poder decirle a quien propone qué tiene disponible para cada
    # uno. Deduplicar no tenía por qué costar la procedencia.
    _de_quien = {}
    for _i, m in enumerate(partes, 1):
        for t in m.tesis:
            _de_quien.setdefault(t["registro"], []).append(_i)
    for m in partes:
        for t in m.tesis:
            if t["registro"] not in r_vistos:
                r_vistos.add(t["registro"])
                t["para"] = sorted(set(_de_quien.get(t["registro"], [])))
                tesis.append(t)
        for n in m.normas:
            clave = (n["cuerpo_legal"], str(n["articulo"]))
            if clave not in n_vistos:
                n_vistos.add(clave)
                normas.append(n)

    # ═══════════════════════════════════════════════════════════════════════
    # ESTE ES EL ORDEN QUE EL MODELO VE DE VERDAD
    # ═══════════════════════════════════════════════════════════════════════
    # `material_para` ordena lo de CADA problema; esto fusiona los problemas y
    # vuelve a ordenar. Arreglé la jerarquía allí y aquí se quedó la vieja: el
    # estudio seguía recibiendo la tesis aislada de la Corte por delante de la
    # jurisprudencia de colegiado. Es el mismo descuido de siempre —dos sitios,
    # arreglado uno—, y aquí duele más porque éste es el último.
    tesis.sort(key=lambda t: (_rango_instancia(t),
                              not t["obligatoria"],
                              -int(t.get("veces") or 0),
                              _de_otro_estado(t, coleccion_estatal)))
    # Y LOS PRINCIPIOS DE TODOS LOS PROBLEMAS, sin repetir: se perdían aquí.
    _pr, _vis = [], set()
    for m in partes:
        for x in (getattr(m, "principios", None) or []):
            if x not in _vis:
                _vis.add(x)
                _pr.append(x)
    return f6.Material(tesis=tesis, normas=normas, principios=_pr[:8])


# ═══════════════════════════════════════════════════════════════════════════
# LAS TESIS DE LA TÉCNICA, TRAÍDAS POR SU REGISTRO
# ═══════════════════════════════════════════════════════════════════════════
# La búsqueda del acervo va detrás de los PROBLEMAS DEL CASO, y hace bien: es
# lo que el secretario necesita para resolver el fondo. Pero hay cuestiones que
# no son del caso sino de la TÉCNICA —si procede el reenvío, si el colegiado
# puede sustituir a la Sala—, y ésas no las va a pedir nadie: no son un
# problema jurídico del expediente, son la regla con la que se escribe el
# resolutivo.
#
# Medido en la revisión fiscal 91/2025 generada: el estudio argumentó el
# reenvío con todas las letras —«No corresponde a este Tribunal Colegiado
# sustituir a la Sala responsable en el estudio de los conceptos de anulación
# que quedaron pendientes»— y no citó NINGUNA autoridad, porque la búsqueda
# había ido detrás de la notificación electrónica. Las cuatro tesis que lo
# sostienen estaban en la colección; nadie las pidió.
#
# Se traen por su número de registro, que es lo contrario de adivinar: o esa
# tesis existe con ese número, o no se trae nada.
async def tesis_por_registro(qdrant, registros: list) -> list:
    """Las tesis con esos registros, tal como están en el acervo."""
    from qdrant_client.models import FieldCondition, Filter, MatchAny
    regs = [str(r).strip() for r in (registros or []) if str(r).strip()]
    if not qdrant or not regs:
        return []
    try:
        r = qdrant.scroll(
            collection_name=COLECCION_JURIS,
            scroll_filter=Filter(must=[FieldCondition(
                key="registro", match=MatchAny(any=regs))]),
            limit=len(regs) * 2, with_payload=True)
        if inspect.isawaitable(r):
            r = await r
        pts = r[0] if isinstance(r, tuple) else r
    except Exception as e:
        print(f"   ⚠️ no se pudieron traer las tesis de la técnica: {e}")
        return []
    fuera, vistos = [], set()
    for p in pts:
        d = _tesis_de(p.payload or {})
        if d["registro"] and d["registro"] not in vistos and d["rubro"]:
            vistos.add(d["registro"])
            # MARCADAS. El prompt sólo admite MAX_TESIS_PROMPT tesis y estas
            # llegaban al final de una lista de cuarenta: el recorte se las
            # llevaba siempre. La marca las exime del tope.
            d["tecnica"] = True
            fuera.append(d)
    return fuera


# ═══════════════════════════════════════════════════════════════════════════
# LO QUE CITAN LAS SENTENCIAS QUE YA RESOLVIERON ESTA MISMA CUESTIÓN
# ═══════════════════════════════════════════════════════════════════════════
# David: «una sentencia se sostiene por su nivel de argumentación jurídica y
# por los criterios VINCULANTES que invoca».
#
# La búsqueda por parecido de rubro es buena encontrando algo PARECIDO y
# estructuralmente mala encontrando lo OBLIGATORIO. Medido sobre las 12
# preguntas del banco de calidad, contando cuántos de los diez criterios
# recuperados vinculan:
#
#     semántica ....... 25 de 120  (21%)
#     co-citación ..... 90 de 115  (78%)
#     de Sala o Pleno .. 37  →  97
#     invisibles para la semántica ......... 104 de 115
#
# Mejora en LAS DOCE preguntas, sin excepción. Y no es un ajuste de pesos: son
# criterios que la semántica no devuelve a ninguna profundidad, porque su rubro
# no se parece a la pregunta aunque sea la autoridad que decide el punto.
#
# DE DÓNDE SALE. El 81% de los 200,650 holdings guarda `tesis_registros`: los
# criterios que esa sentencia citó de verdad. Contando cuáles se repiten en las
# sentencias que resolvieron la misma cuestión sale, sin modelo y sin
# adivinanza, qué autoridad usa el circuito para ese punto —y cuántas veces,
# que es un dato que la semántica no puede dar—.
#
# LA TRAMPA: `tesis_registros` mezcla DOS identificadores, el registro digital
# («2016701») y la clave de la tesis («2a./J. 137/2016 (10a.)»). Por eso se
# busca por los dos campos. Aun así, el 39% de las claves no resuelve contra la
# colección: son citas que no podemos convertir en autoridad utilizable, y eso
# es trabajo pendiente, no un fallo de esto.
COLECCION_SENTENCIAS = "sentencias_holdings"
SENTENCIAS_POR_PROBLEMA = 50
COCITADAS_POR_PROBLEMA = 8


# LA MISMA TESIS SE ESCRIBE DE VARIAS MANERAS, y por eso no casa. Medido sobre
# 300 claves reales: con la búsqueda exacta resuelven 163 (54%). Lo que falla
# no es que falte corpus —eso creí al principio— sino la ortografía de la cita:
#
#     «2a./J.81/2002»      sin el espacio detrás de «J.»
#     «VI.2º. J/129»       con el ordinal masculino en vez de «2o.»
#     «2a./3. 115/2005»    con la «J» que el OCR leyó como un tres
#     «XVII.1o.C.T.30K»    sin el espacio antes de la letra final
#
# Probando esas variantes se pasa de 163 a 181 (60%). **Es una ganancia
# modesta, no la mitad que yo había estimado**: el 40% restante son criterios
# que de verdad no están en la colección —Séptima y Octava Época, precedentes
# del propio Tribunal Federal de Justicia Administrativa— y ésos no se
# resuelven normalizando, sino ingiriéndolos.
_RX_ORDINAL_MASC = re.compile(r"[º°]")


def _variantes_de_clave(k: str) -> list:
    """Las formas en que la misma clave aparece escrita, la primera la literal."""
    k = " ".join(str(k or "").split())
    if not k:
        return []
    v = [k]
    a = _RX_ORDINAL_MASC.sub("o.", k).replace("o..", "o.")
    if a != k:
        v.append(a)
    for x in list(v):
        for y in (re.sub(r"(J\.)(\d)", r"\1 \2", x),          # J.81 → J. 81
                  re.sub(r"(J\.)\s+(\d)", r"\1\2", x),         # J. 81 → J.81
                  re.sub(r"\s*\(\d+a\.\)\s*$", "", x),        # sin la época
                  re.sub(r"(\d)([A-Z])$", r"\1 \2", x),        # 30K → 30 K
                  x.replace("/3.", "/J.").replace("/1.", "/J.")):
            if y and y != x:
                v.append(y)
    fuera = []
    for x in v:
        x = " ".join(x.split())
        if x and x not in fuera:
            fuera.append(x)
    return fuera[:10]


async def _ficha_de_cita(qdrant, clave: str):
    """La tesis, por registro y por su clave, tolerando cómo se escribió."""
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    async def _uno(campo, val):
        try:
            r = qdrant.scroll(
                collection_name=COLECCION_JURIS,
                scroll_filter=Filter(must=[FieldCondition(
                    key=campo, match=MatchValue(value=str(val)))]),
                limit=1, with_payload=True)
            if inspect.isawaitable(r):
                r = await r
            pts = r[0] if isinstance(r, tuple) else r
            return _tesis_de(pts[0].payload or {}) if pts else None
        except Exception:
            return None

    # El registro, con o sin el «Registro digital» delante.
    m = re.match(r"^\D*(\d{5,8})\D*$", str(clave))
    if m:
        f = await _uno("registro", m.group(1))
        if f:
            return f
    for var in _variantes_de_clave(clave):
        f = await _uno("clave_tesis", var)
        if f:
            return f
    return None


async def tesis_cocitadas(qdrant, embed_leyes, problema: str,
                          tope: int = COCITADAS_POR_PROBLEMA) -> list:
    """Los criterios que más cita el circuito al resolver esta cuestión."""
    if not qdrant or not (problema or "").strip():
        return []
    try:
        v = await embed_leyes(problema)
        r = qdrant.query_points(
            collection_name=COLECCION_SENTENCIAS, query=v, using="dense",
            limit=SENTENCIAS_POR_PROBLEMA,
            # LOS PRINCIPIOS VIENEN EN EL MISMO VIAJE. Están en el 99.3% de los
            # holdings y no cuestan una consulta más: son las nociones con las
            # que el circuito razona esta cuestión —«principio de literalidad
            # de los títulos de crédito», «interpretación restrictiva de la
            # prescripción»— y decirle al modelo cuáles usa el tribunal es
            # decirle por dónde va el razonamiento, no sólo qué citar.
            with_payload=["tesis_registros", "principios_juridicos"])
        if inspect.isawaitable(r):
            r = await r
        pts = getattr(r, "points", r)
    except Exception as e:
        print(f"   ⚠️ co-citación: no se pudo consultar el acervo: {e}")
        return []
    from collections import Counter
    cuenta = Counter()
    principios = Counter()
    for p in pts:
        _pl = p.payload or {}
        for x in (_pl.get("tesis_registros") or []):
            if str(x).strip():
                cuenta[str(x).strip()] += 1
        for x in (_pl.get("principios_juridicos") or []):
            if str(x).strip():
                principios[" ".join(str(x).split()).lower()] += 1
    # Los principios se devuelven aparte, por referencia, para que quien llame
    # pueda usarlos aunque no haya ninguna tesis que resolver.
    tesis_cocitadas.ultimos_principios = [
        p for p, n in principios.most_common(6) if n >= 2]
    if not cuenta:
        return []
    # SÓLO LAS QUE SE REPITEN MANDAN ARRIBA, pero una sola cita también vale:
    # en cuestiones poco litigadas puede no haber más. Se piden las 20 más
    # citadas y se resuelven a la vez; se devuelven las `tope` que existan.
    claves = [k for k, _ in cuenta.most_common(20)]
    fichas = await asyncio.gather(*[_ficha_de_cita(qdrant, k) for k in claves])
    fuera = []
    for k, f in zip(claves, fichas):
        if f and f.get("registro") and f.get("rubro"):
            f["veces"] = cuenta[k]
            f["cocitada"] = True
            fuera.append(f)
        if len(fuera) >= tope:
            break
    if fuera:
        print(f"   ⚖️ co-citación: {len(fuera)} criterios usados por el circuito "
              f"(el más citado, {fuera[0].get('veces')} veces)")
    return fuera


# ═══════════════════════════════════════════════════════════════════════════
# LAS TESIS DE LA CALIFICATIVA
#
# David: «resulta particularmente importante que el RAG recupere tesis sobre
# INOPERANCIA (una pequeña llamada adicional) para que, si el secretario se va
# por inoperancia, el LLM proponga argumentos con respaldo en tesis sobre
# inoperancia».
#
# Tiene razón y el hueco es estructural: la búsqueda del acervo va detrás de
# los PROBLEMAS DEL CASO —la pericial, la notificación, el despido— y eso está
# bien para el fondo. Pero declarar un planteamiento INOPERANTE no es una
# cuestión de fondo: es una cuestión de TÉCNICA, y se funda con tesis sobre la
# inoperancia misma —que no combate la consideración toral, que parte de una
# premisa falsa, que es una repetición de la demanda—. Ninguna búsqueda del
# caso las va a traer.
#
# Se vio en los proyectos generados: el estudio declaraba inoperante y no
# citaba una sola autoridad, o citaba la del fondo, que no viene al caso.
#
# Es una llamada pequeña y sólo se hace cuando el secretario elige una de esas
# calificativas: si resuelve el fondo, no hace falta.
# ═══════════════════════════════════════════════════════════════════════════

# Qué se busca para cada calificativa. No son sinónimos y sus tesis son
# distintas: la inoperancia mira si el planteamiento SIRVE, la ineficacia si
# ALCANZA, y lo inatendible si puede siquiera examinarse.
CONSULTA_POR_CALIFICATIVA = {
    "inoperante": (
        "agravios inoperantes conceptos de violación inoperantes porque no "
        "combaten todas las consideraciones de la sentencia recurrida, se "
        "sustentan en premisas falsas, repiten los argumentos de la demanda o "
        "no controvierten la consideración toral que rige el sentido"),
    "inatendible": (
        "conceptos de violación inatendibles planteamientos que no pueden "
        "examinarse por referirse a cuestiones ajenas a la litis o a actos "
        "consentidos"),
    "ineficaz": (
        "agravios ineficaces para revocar la sentencia porque aun siendo "
        "fundados no trascienden al resultado del fallo ni bastan para variar "
        "el sentido de lo resuelto"),
    "innecesario": (
        "estudio innecesario de los conceptos restantes por haberse alcanzado "
        "el beneficio máximo con la concesión del amparo, sustracción de la "
        "materia"),
    "fundado_insuficiente": (
        "agravio fundado pero insuficiente para revocar porque subsisten "
        "consideraciones que rigen el sentido del fallo y no fueron combatidas"),
}


async def tesis_de_la_calificativa(qdrant, embed_juris, sentido: str,
                                   problema: str = "", limite: int = 4) -> list:
    """Las tesis que fundan ESA calificativa, no el fondo del asunto.

    Devuelve [] cuando la calificativa resuelve el fondo —fundado, infundado—:
    para ésas ya sirve el material del caso, y añadir tesis de técnica sólo
    haría ruido.
    """
    clave = (sentido or "").strip().lower()
    consulta = CONSULTA_POR_CALIFICATIVA.get(clave)
    if not consulta or not qdrant or not embed_juris:
        return []
    # El problema se añade recortado y al final: orienta hacia la materia sin
    # desplazar el concepto, que es lo que esta búsqueda tiene que encontrar.
    _p = " ".join((problema or "").split())[:180]
    try:
        v = await embed_juris(consulta + (" " + _p if _p else ""))
        res = await _buscar(qdrant, COLECCION_JURIS, VECTOR_RUBRO, v, limite * 3)
    except Exception as e:
        print(f"   ⚠️ no se pudieron traer las tesis de «{clave}»: {e}")
        return []
    fuera, vistos = [], set()
    for p in res:
        d = _tesis_de(p if isinstance(p, dict) else (p.payload or {}))
        if d.get("registro") and d["registro"] not in vistos and d.get("rubro"):
            vistos.add(d["registro"])
            # Marcadas como técnica: quedan exentas del tope del prompt, igual
            # que las del reenvío. Son pocas y son las que fundan el fallo.
            d["tecnica"] = True
            d["de_la_calificativa"] = clave
            fuera.append(d)
        if len(fuera) >= limite:
            break
    return fuera
