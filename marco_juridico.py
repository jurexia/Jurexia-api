"""EL MARCO JURÍDICO: los preceptos aplicables, con su texto, antes de resolver.

Lo pidió David así: «antes de resolver los problemas jurídicos, sentar un marco
jurídico en el que, vía RAG, bajes los artículos constitucionales como el
primero y el cuarto, los de fuente convencional que resulten aplicables, o
precedentes de los cuadernillos de la Corte Interamericana. Falta incrementar
marco jurídico con la CITA TEXTUAL de los artículos, incluyendo los aplicables
del Estado de Querétaro.»

LO QUE HAY EN EL ACERVO (`bloque_constitucional`, 6,823 puntos):
    cuadernillos CoIDH .... 5,212  (24 cuadernillos, con caso y párrafo)
    convenciones ..........   950
    constitución ..........   355
    sentencias CoIDH ......   296
    opiniones consultivas .    10

DOS TRAMPAS QUE COSTARÍAN UNA CITA FALSA
════════════════════════════════════════

1. EL CAMPO `ref` MIENTE. 84 de los 355 fragmentos constitucionales dicen
   `ref='CPEUM · Transitorios de reformas'` cuando su `jerarquia` dice
   `Art. 19 CPEUM (parte 4)`. Buscar el artículo 16 por `ref` devolvía un
   transitorio de 1917 sobre el período de sesiones del Congreso. **La fuente de
   verdad es `jerarquia`**, que trae la ruta completa CPEUM > TÍTULO > Art. N.

2. LOS ARTÍCULOS VIENEN TROCEADOS: el 2º en 19 partes, el 4º en 8, el 123 en 8.
   Citar «textualmente» un artículo exige REUNIR sus partes en orden; quedarse
   con el trozo que devolvió la búsqueda es citar un párrafo suelto y llamarlo
   artículo.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Optional

COLECCION = "bloque_constitucional"

# Cuántas piezas entran al marco. Medido el problema contrario: el sistema ya
# escribe entre 25% y 60% más que el secretario, así que un marco generoso
# empeoraría lo que peor está. Pocas y pertinentes.
MAX_CONSTITUCIONALES = 3
MAX_CONVENCIONALES = 2
MAX_COIDH = 2
MAX_LOCALES = 3

_RX_ART_JERARQUIA = re.compile(r"Art\.?\s*(\d{1,3})\s*[o°ºª]?\s*(?:BIS|TER)?\s*CPEUM"
                               r"(?:\s*\(parte\s*(\d+)\))?", re.I)

# LA BÚSQUEDA SEMÁNTICA NO SIRVE PARA LOCALIZAR UN ARTÍCULO CONSTITUCIONAL.
# Probado: para «derecho de habitación de menores en una copropiedad» devolvió
# el artículo 27 —tierras y aguas de la Nación— y el 18 —sistema penitenciario—.
# Los artículos son largos y genéricos, y el vector de un problema concreto no
# los distingue.
#
# Se usa un MAPA TEMÁTICO, que no es un marco fijo: sólo entra el artículo cuyo
# tema aparece en los problemas de ESTE asunto. Si ninguno aparece, no entra
# ninguno, que es lo que David pidió — «sobre la solución en función del
# problema jurídico», no una plantilla pegada en todos.
TEMAS_CONSTITUCIONALES = {
    "1": ("derechos humanos", "pro persona", "interpretación conforme",
          "control de convencionalidad", "discriminación", "convencionalidad"),
    "4": ("interés superior", "menor", "menores", "niña", "niño", "adolescente",
          "familia", "alimentos", "habitación", "igualdad entre el hombre y la mujer",
          "salud", "vivienda digna"),
    "14": ("retroactividad", "formalidades esenciales", "debido proceso",
           "privación", "audiencia", "exacta aplicación"),
    "16": ("fundamentación", "motivación", "acto de molestia", "mandamiento escrito"),
    "17": ("acceso a la justicia", "tutela judicial", "justicia pronta",
           "recurso efectivo", "gratuidad"),
    "27": ("propiedad de tierras y aguas", "expropiación", "dominio de la nación"),
    "31": ("proporcionalidad y equidad", "legalidad tributaria", "contribuciones"),
    "123": ("relación de trabajo", "salario", "jornada", "despido", "trabajador"),
}

# «RECORRIÓ EN SU ORDEN PARA SER UN NUEVO DÉCIMO PÁRRAFO» no es texto del
# artículo: es la nota de una reforma, y entraba al proyecto como si fuera ley.
_RX_NOTA_REFORMA = re.compile(
    r"^\s*(?:SE\s+|PÁRRAFO\s+|FRACCI[ÓO]N\s+)?(?:RECORRI[ÓO]|ADICIONAD|REFORMAD|"
    r"DEROGAD|FE\s+DE\s+ERRATAS)", re.I)


# La capa convencional entra SÓLO si el problema la exige. Sin este filtro, la
# Convención Americana se colaba en una revisión fiscal sobre cuota pensionaria:
# ruido con aspecto de erudición, que es la peor clase de relleno. Medido: el
# secretario la usa en 19 de 125 proyectos, y la mayoría son recitaciones de lo
# que alegó el quejoso, no voz propia.
TEMAS_CONVENCIONALES = (
    "derechos humanos", "interés superior", "menor", "menores", "niña", "niño",
    "adolescente", "convencionalidad", "pro persona", "discriminación",
    "recurso efectivo", "acceso a la justicia", "tutela judicial", "usura",
    "perspectiva de género", "igualdad", "vida digna", "tortura", "desaparición",
    "libertad de expresión", "pueblos indígenas", "propiedad colectiva",
)


def _pide_convencional(problemas: list[str]) -> bool:
    t = " ".join(problemas or []).lower()
    return any(c in t for c in TEMAS_CONVENCIONALES)


def _articulos_del_problema(problemas: list[str]) -> list[str]:
    """Los artículos constitucionales que ESTE asunto toca, por su tema."""
    texto = " ".join(problemas or []).lower()
    fuera = []
    for art, claves in TEMAS_CONSTITUCIONALES.items():
        if any(c in texto for c in claves):
            fuera.append(art)
    return fuera


@dataclass
class Precepto:
    """Un artículo con su texto ÍNTEGRO, reunido de todas sus partes."""
    fuente: str          # «Constitución», «Convención sobre los Derechos del Niño»…
    articulo: str
    texto: str
    jerarquia: str = ""
    orden: int = 0       # para presentarlos por jerarquía normativa


@dataclass
class Precedente:
    """Un párrafo de la Corte Interamericana, con su caso."""
    caso: str
    cuadernillo: str
    tema: str
    parrafo: str
    texto: str


@dataclass
class Marco:
    constitucionales: list[Precepto] = field(default_factory=list)
    convencionales: list[Precepto] = field(default_factory=list)
    locales: list[Precepto] = field(default_factory=list)
    coidh: list[Precedente] = field(default_factory=list)
    avisos: list[str] = field(default_factory=list)

    def vacio(self) -> bool:
        return not (self.constitucionales or self.convencionales
                    or self.locales or self.coidh)


def _articulo_de(payload: dict) -> tuple[str, int]:
    """(número de artículo, número de parte) leídos de `jerarquia`, no de `ref`."""
    j = str(payload.get("jerarquia") or "")
    m = _RX_ART_JERARQUIA.search(j)
    if m:
        return m.group(1), int(m.group(2) or 1)
    # Sin jerarquía utilizable, se acepta `ref` con reservas.
    m2 = _RX_ART_JERARQUIA.search(str(payload.get("ref") or ""))
    return (m2.group(1), int(m2.group(2) or 1)) if m2 else ("", 1)


def _es_transitorio(payload: dict) -> bool:
    j = (str(payload.get("jerarquia") or "") + " "
         + str(payload.get("ref") or "")).lower()
    # Sólo cuenta como transitorio si NO se puede resolver un artículo del
    # articulado permanente: el `ref` está mal poblado y por sí solo descartaría
    # artículos buenos.
    return "transitorio" in j and not _articulo_de(payload)[0]


async def _buscar(qdrant, coleccion: str, vector: str, v: list[float],
                  limite: int, filtro=None) -> list[dict]:
    import inspect
    try:
        r = qdrant.query_points(collection_name=coleccion, query=v, using=vector,
                                limit=limite, query_filter=filtro, with_payload=True)
        if inspect.isawaitable(r):
            r = await r
        return [p.payload or {} for p in r.points]
    except Exception:
        return []


def _reunir_articulo(fragmentos: list[dict]) -> str:
    """Todas las partes de un artículo, en orden y sin repetir."""
    partes = sorted(fragmentos, key=lambda p: _articulo_de(p)[1])
    fuera, visto = [], set()
    for p in partes:
        t = (p.get("texto") or p.get("texto_visible") or "").strip()
        if _RX_NOTA_REFORMA.match(t):      # notas de reforma, no articulado
            continue
        if t and t[:80] not in visto:
            visto.add(t[:80])
            fuera.append(t)
    return "\n".join(fuera)


async def construir(qdrant, embed, problemas: list[str],
                    coleccion_estatal: Optional[str] = None) -> Marco:
    """El marco de ESTE asunto, no una plantilla de derechos humanos.

    Medido sobre 125 proyectos del secretario: la Convención sobre los Derechos
    del Niño aparece en UNO, el artículo 4º constitucional en siete y la Corte
    Interamericana en seis. Lo que domina es la Ley de Amparo, la jurisprudencia
    de la Corte y el código local. Y la regla que él sigue: **el marco arranca
    por la naturaleza de la figura jurídica discutida, no por los derechos
    humanos**.

    Por eso aquí no se pega nada por defecto: se busca lo que los problemas
    piden, y si no hay nada pertinente el marco se queda vacío y no se escribe.
    """
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    m = Marco()
    consulta = " ".join(p for p in problemas if p)[:900]
    if not consulta.strip():
        return m
    v = await embed(consulta)

    def _f(tipo):
        return Filter(must=[FieldCondition(key="tipo", match=MatchValue(value=tipo))])

    # Los constitucionales NO se buscan por vector: se piden por número, según
    # el tema que los problemas mencionen.
    arts_pedidos = _articulos_del_problema(problemas)
    quiere_conv = _pide_convencional(problemas)
    const, conv, coidh, locales = await asyncio.gather(
        _buscar(qdrant, COLECCION, "dense", v, 60, _f("constitucion"))
        if arts_pedidos else _vacio(),
        _buscar(qdrant, COLECCION, "dense", v, 8, _f("convencion"))
        if quiere_conv else _vacio(),
        _buscar(qdrant, COLECCION, "dense", v, 6, _f("cuadernillo"))
        if quiere_conv else _vacio(),
        _buscar(qdrant, coleccion_estatal, "dense", v, 8) if coleccion_estatal else _vacio(),
    )

    # Constitucionales: se agrupan por artículo y se reúnen sus partes, porque
    # vienen troceados —el 2º en 19 pedazos— y citar un trozo no es citar el
    # artículo.
    por_art: dict = {}
    for p in const:
        if _es_transitorio(p):
            continue
        art = _articulo_de(p)[0]
        if art and art in arts_pedidos:
            por_art.setdefault(art, []).append(p)
    for art in [a for a in arts_pedidos if a in por_art][:MAX_CONSTITUCIONALES]:
        texto = _reunir_articulo(por_art[art])
        if texto:
            m.constitucionales.append(Precepto(
                fuente="Constitución Política de los Estados Unidos Mexicanos",
                articulo=f"{art}o.", texto=texto,
                jerarquia=str(por_art[art][0].get("jerarquia") or ""), orden=1))

    for p in conv[:MAX_CONVENCIONALES]:
        t = (p.get("texto") or "").strip()
        if t:
            m.convencionales.append(Precepto(
                fuente=str(p.get("origen") or p.get("ref") or "Tratado internacional"),
                articulo=str(p.get("ref") or ""), texto=t, orden=2))

    for p in coidh:
        if len(m.coidh) >= MAX_COIDH:
            break
        t = (p.get("texto") or "").strip()
        # Sin caso ni cuadernillo identificable no se cita: un párrafo de la
        # Corte Interamericana sin su fuente no se puede comprobar, y una cita
        # que no se comprueba no debería entrar en una sentencia.
        if t and (p.get("caso") or p.get("vs") or p.get("cuadernillo_num")):
            m.coidh.append(Precedente(
                caso=str(p.get("caso") or p.get("vs") or ""),
                cuadernillo=str(p.get("cuadernillo_num") or ""),
                tema=str(p.get("cuadernillo_tema") or ""),
                parrafo=str(p.get("parrafo") or ""), texto=t))

    for p in (locales or [])[:MAX_LOCALES]:
        t = (p.get("texto") or "").strip()
        if t:
            m.locales.append(Precepto(
                fuente=str(p.get("cuerpo_legal_oficial") or p.get("ref") or ""),
                articulo=str(p.get("articulo_num") or ""), texto=t, orden=3))
    return m


async def _vacio():
    return []


def bloque(m: Marco, es_recurso: bool = False) -> str:
    """Lo que se le pone al redactor para que construya el marco.

    NO es el marco escrito: son los materiales. Lo redacta él, en su prosa, y
    sólo con lo que de verdad aplique. Si algo de aquí no viene al caso, se
    calla: un marco con piezas de adorno es peor que no tenerlo.
    """
    if m.vacio():
        return ""
    q = "agravios" if es_recurso else "conceptos de violación"
    p = ["", "═" * 71, "MATERIALES PARA EL MARCO JURÍDICO", "═" * 71,
         "Van AQUÍ los preceptos que el acervo encontró para los problemas de",
         "este asunto. Con ellos se escribe el marco, con estas reglas medidas",
         "sobre 125 engroses de este tribunal:",
         "",
         "  · EL MARCO ARRANCA POR LA FIGURA JURÍDICA discutida —la acción, la",
         "    prestación, el presupuesto procesal—, NO por los derechos humanos.",
         "  · VA DESPUÉS de anunciar el sentido y ANTES del caso concreto.",
         "  · Se TRANSCRIBE literalmente el precepto LOCAL o secundario decisivo,",
         "    entre comillas y con el número de artículo al frente.",
         "  · La CONSTITUCIÓN se PARAFRASEA, no se transcribe: «El artículo 4º",
         "    constitucional reconoce el derecho a…». Es lo que él hace.",
         "  · La capa CONVENCIONAL y la Corte Interamericana entran SÓLO si el",
         "    problema las exige. Si no vienen al caso, NO SE PONEN.",
         "  · EXTENSIÓN: entre 600 y 1,200 palabras. Más corto en familia, donde",
         "    el peso está en la prueba y no en la norma.",
         "  · CIERRA CON UNA BISAGRA que devuelva al expediente antes de entrar al",
         f"    caso: sin ella el marco queda flotando y no contesta los {q}.",
         ""]
    if m.locales:
        p.append("── PRECEPTOS LOCALES Y SECUNDARIOS (éstos SÍ se transcriben) ──")
        for x in m.locales:
            p.append(f"\n  {x.fuente} — Artículo {x.articulo}")
            p.append(f"  «{x.texto[:1400]}»")
    if m.constitucionales:
        p.append("\n── CONSTITUCIONALES (se parafrasean) ──")
        for x in m.constitucionales:
            p.append(f"\n  Artículo {x.articulo} de la Constitución")
            p.append(f"  {x.texto[:1200]}")
    if m.convencionales:
        p.append("\n── CONVENCIONALES (sólo si el problema los exige) ──")
        for x in m.convencionales:
            p.append(f"\n  {x.fuente} · {x.articulo}")
            p.append(f"  {x.texto[:900]}")
    if m.coidh:
        p.append("\n── CORTE INTERAMERICANA (sólo si el problema la exige) ──")
        for x in m.coidh:
            ficha = f"Caso {x.caso}" if x.caso else f"Cuadernillo {x.cuadernillo}"
            if x.parrafo:
                ficha += f", párr. {x.parrafo}"
            p.append(f"\n  {ficha} — {x.tema}")
            p.append(f"  {x.texto[:900]}")
    return "\n".join(p)
