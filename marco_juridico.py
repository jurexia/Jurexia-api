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
    # RAÍCES, NO PALABRAS ENTERAS. El problema jurídico del ADC 380/2025 decía
    # «la pensión alimenticia definitiva en favor de E.M.O.R.»: no contiene
    # «alimentos» ni «menor» —la menor va por sus iniciales— y el marco no se
    # construyó en un asunto de familia con una niña de por medio. Con «aliment»
    # entra alimentos, alimenticia y alimentaria.
    "4": ("interés superior", "menor", "menores", "niña", "niño", "adolescente",
          "infancia", "familia", "aliment", "orfandad", "pensión", "habitación",
          "custodia", "guarda", "convivencia", "patria potestad", "filiación",
          "igualdad entre el hombre y la mujer", "salud", "vivienda digna"),
    "14": ("retroactividad", "formalidades esenciales", "debido proceso",
           "privación", "audiencia", "exacta aplicación"),
    "16": ("fundamentación", "motivación", "acto de molestia", "mandamiento escrito"),
    "17": ("acceso a la justicia", "tutela judicial", "justicia pronta",
           "recurso efectivo", "gratuidad"),
    "27": ("propiedad de tierras y aguas", "expropiación", "dominio de la nación"),
    # «proporcionalidad y equidad» a secas disparaba el 31 —que es TRIBUTARIO—
    # en un asunto de alimentos, donde «proporcionalidad» es la del artículo
    # 296 del código civil. Un artículo de más en el marco no es neutro: ocupa
    # una de las tres plazas y deja fuera al que sí venía al caso.
    "31": ("proporcionalidad y equidad tributaria", "legalidad tributaria",
           "contribuciones", "gasto público", "impuesto"),
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


# ARTÍCULOS QUE ABREN EL BLOQUE. Si el asunto toca uno de éstos, la fuente
# convencional y la Corte Interamericana son pertinentes: son los derechos
# cuyo contenido está también en los tratados.
ARTS_DE_BLOQUE = ("1", "4", "14", "17")


def _pide_convencional(problemas: list[str]) -> bool:
    """Si el asunto llama al bloque de constitucionalidad.

    NO basta con buscar palabras sueltas. Medido en el ADC 380/2025: los
    problemas decían «pensión alimenticia definitiva en favor de E.M.O.R.» —la
    menor va por sus iniciales— y ninguna clave casaba, así que la capa
    convencional NO SE BUSCÓ NUNCA. El estudio acabó nombrando la Convención
    sobre los Derechos del Niño seis veces de memoria del modelo, sin un solo
    fragmento del acervo detrás. Una cita que nadie puede comprobar es
    exactamente lo que este sistema existe para evitar.

    La regla buena: si el mapa temático disparó un artículo del bloque, la
    fuente convencional viene al caso.
    """
    t = " ".join(problemas or []).lower()
    if any(c in t for c in TEMAS_CONVENCIONALES):
        return True
    return any(a in _arts_base(problemas) for a in ARTS_DE_BLOQUE)


def _arts_base(problemas: list[str]) -> list[str]:
    """Los artículos por tema, sin la puerta del 1º: rompe la circularidad."""
    texto = " ".join(problemas or []).lower()
    return [art for art, claves in TEMAS_CONSTITUCIONALES.items()
            if any(c in texto for c in claves)]


def _articulos_del_problema(problemas: list[str]) -> list[str]:
    """Los artículos constitucionales que ESTE asunto toca, por su tema."""
    fuera = list(_arts_base(problemas))
    # EL 1º ES LA PUERTA DEL BLOQUE. Si el asunto llama a fuente convencional o
    # a la Corte Interamericana, el artículo que permite aplicarlas en México
    # es el 1º: sin él la cita de un tratado queda sin anclaje constitucional.
    # David lo pidió así —«sería bueno el 1 y 4 de la constitución y preceptos
    # convencionales aplicables»— y el asunto de alimentos sólo disparaba el 4º.
    if "1" not in fuera and _pide_convencional(problemas):
        fuera.insert(0, "1")
    return fuera


# LA CONSULTA CONVENCIONAL NO SE HACE CON LA PROSA DEL CASO. Medido: el
# Cuadernillo No. 5 de la CoIDH —«Niños, Niñas y Adolescentes»— tiene 424
# fragmentos en el acervo y no salía ni uno, porque se buscaba con «¿la pensión
# alimenticia del quince por ciento de los ingresos…?» y eso no casa con la
# doctrina de infancia. Es la misma lección que el RAG de jurisprudencia: la
# pregunta conceptual encuentra, la prosa del expediente no.
CONSULTA_POR_ARTICULO = {
    "1": "principio pro persona, interpretación conforme, control de "
         "convencionalidad, obligación de promover, respetar, proteger y "
         "garantizar los derechos humanos",
    "4": "interés superior del niño, derechos de niñas, niños y adolescentes, "
         "obligación reforzada de protección de la infancia, derecho a "
         "alimentos y a un nivel de vida adecuado, deberes de los progenitores",
    "14": "debido proceso, formalidades esenciales del procedimiento, derecho "
          "de audiencia y defensa",
    "16": "fundamentación y motivación de los actos de autoridad",
    "17": "acceso a la justicia, tutela judicial efectiva, recurso sencillo y "
          "efectivo, plazo razonable",
    "123": "derechos laborales, condiciones equitativas y satisfactorias de "
           "trabajo",
}


def _consulta_convencional(arts: list[str], problemas: list[str]) -> str:
    """Lo que se le pregunta al bloque de constitucionalidad.

    Se construye con los CONCEPTOS de los artículos que el asunto disparó, no
    con el relato del expediente.
    """
    piezas = [CONSULTA_POR_ARTICULO[a] for a in arts if a in CONSULTA_POR_ARTICULO]
    if not piezas:
        piezas = [" ".join(problemas or [])[:400]]
    return " ".join(piezas)[:900]


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


def _reunir_articulo(fragmentos: list) -> str:
    """El artículo entero… PERO SÓLO EL QUE ES.

    En la Constitución hay DOS artículos 17: el del Título Primero —«Ninguna
    persona podrá hacerse justicia por sí misma»— y el del Título Noveno, sobre
    la inviolabilidad de la Constitución, que empieza «Los Templos y demás
    bienes». El acervo los trae los dos con `articulo_num = 17` y esta función
    los pegaba en un solo texto, así que el proyecto transcribía como artículo
    17 constitucional un Frankenstein de dos artículos distintos.

    Los fragmentos del MISMO artículo comparten título y capítulo. Se agrupan
    por ahí y se conserva el grupo mayor, que es el articulado real; el otro es
    casi siempre una disposición aislada de un título lejano.
    """
    if not fragmentos:
        return ""
    grupos: dict = {}
    for p in fragmentos:
        clave = (str(p.get("titulo") or ""), str(p.get("capitulo") or ""))
        grupos.setdefault(clave, []).append(p)
    # El grupo con más fragmentos; a empate, el del título más temprano, que en
    # la Constitución es el de las garantías.
    elegidos = max(grupos.values(), key=lambda g: (len(g), -_orden_titulo(g[0])))
    elegidos = sorted(elegidos, key=lambda p: int(p.get("chunk_index") or 0))
    partes, visto = [], set()
    for p in elegidos:
        x = " ".join(str(p.get("texto") or "").split())
        if x and x not in visto:
            visto.add(x)
            partes.append(x)
    return " ".join(partes)


_ORD_TITULO = ["primero", "segundo", "tercero", "cuarto", "quinto", "sexto",
               "septimo", "octavo", "noveno", "decimo"]


def _orden_titulo(p: dict) -> int:
    t = str(p.get("titulo") or "").lower()
    for i, w in enumerate(_ORD_TITULO):
        if w in t or w.replace("septimo", "séptimo") in t:
            return i
    return 99

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
