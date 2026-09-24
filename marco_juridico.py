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
           "privación", "audiencia", "exacta aplicación",
           # «LEGALIDAD Y SEGURIDAD JURÍDICA» es la fórmula con que el amparo
           # nombra los artículos 14 y 16, y el mapa no la conocía. Medido el
           # 24-sep-2026: el juzgado del 711/2025 resolvió que el bloqueo
           # «vulneró los derechos de legalidad y seguridad jurídica… sin
           # determinación fundada y motivada», y el marco salió VACÍO.
           "seguridad jurídica"),
    "16": ("fundamentación", "motivación", "acto de molestia", "mandamiento escrito",
           "legalidad", "seguridad jurídica", "motivad", "fundar y motivar"),
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


# LO GENÉRICO, DETRÁS. «Legalidad y seguridad jurídica» aparece en casi todo
# amparo, y el cupo es de tres: si el 14 y el 16 van delante, se comen el
# artículo que de verdad es del asunto —el 4º en familia, el 123 en trabajo,
# el 31 en lo fiscal—. Se ordenan después de los específicos.
_ARTS_GENERICOS = ("14", "16")


def _articulos_del_problema(problemas: list[str]) -> list[str]:
    """Los artículos constitucionales que ESTE asunto toca, por su tema."""
    base = list(_arts_base(problemas))
    fuera = ([a for a in base if a not in _ARTS_GENERICOS]
             + [a for a in base if a in _ARTS_GENERICOS])
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
    # CON LOS ARTÍCULOS 8.1 Y 25 NOMBRADOS. Son las garantías judiciales y la
    # protección judicial de la Convención Americana, y sin nombrarlos la
    # consulta devolvía doctrina de plazo razonable en materia penal. Y el
    # tercer párrafo del 17 —privilegiar la solución sobre los formalismos—,
    # que es el que decide las puertas procesales (24-sep-2026).
    "17": "acceso a la justicia, tutela judicial efectiva, recurso sencillo y "
          "efectivo, plazo razonable, garantías judiciales y protección judicial "
          "de los artículos 8.1 y 25 de la Convención Americana, privilegiar la "
          "solución del conflicto sobre los formalismos procedimentales",
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
    """Un párrafo de la Corte Interamericana, con SU CASO.

    David: «las citas de los cuadernillos deben tener un caso de referencia —así
    funciona la cita de jurisprudencia de la Corte Interamericana—. Los
    cuadernillos sólo son eso, cuadernillos; lo que importa es lo que contienen,
    sus casos y lo que se resolvió».

    Tiene razón y aquí había DOS fallos encadenados. `construir` rellenaba
    `caso` con el `ref` del fragmento cuando el caso faltaba, y `bloque` escribía
    «Caso » delante de lo que hubiera: de ahí salió «el Caso CoIDH, Cuadernillo
    No. 2, párr. 69», que no es una cita de nada.

    Ahora un `Precedente` SIN CASO NO EXISTE. Medido: 2,202 de los 5,518
    fragmentos de la Corte traen caso (39%), y con ellos la cita se forma
    entera —«Caso Cantoral Benavides Vs. Perú, párr. 84 (Serie C No. 69110)»—.
    El 61% restante es prosa del cuadernillo sin caso identificado: material de
    lectura, no jurisprudencia que se cite.
    """
    caso: str
    vs: str
    cuadernillo: str
    tema: str
    parrafo: str
    texto: str
    serie: str = ""

    def cita(self) -> str:
        """Como se cita la jurisprudencia de la Corte Interamericana."""
        c = f"Caso {self.caso}"
        if self.vs:
            c += f" Vs. {self.vs}"
        if self.parrafo:
            c += f", párr. {self.parrafo}"
        if self.serie:
            c += f" (Serie C No. {self.serie})"
        return c


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

    # MANDA LO QUE EL ACERVO REPITE, NO EL GRUPO MÁS GORDO.
    # «El grupo con más fragmentos» se equivoca cuando el artículo bueno está
    # partido en varios títulos y el intruso no. Medido el 16-sep-2026: el
    # artículo 16 constitucional vive DOS veces en el acervo —Título Primero,
    # capítulos I y IV, con el mismo texto: «Nadie puede ser molestado en su
    # persona, familia, domicilio…»— y el 16 TRANSITORIO de 1917 vive una,
    # bajo «TITULO NOVENO.». Tres grupos de un fragmento cada uno: ganaba el
    # transitorio por el desempate del título, y se firmó en la revisión
    # fiscal 2/2026 una nota al pie que decía «Artículo 16» y transcribía el
    # Congreso Constituyente convocando al Congreso de la Unión.
    #
    # Dos grupos que empiezan igual son el MISMO artículo contado dos veces, y
    # eso es corroboración: el acervo repite lo que de verdad es el artículo y
    # el intruso aparece una sola vez. Se suman los grupos que coinciden y
    # gana el más corroborado; a empate, el que traiga más fragmentos, y a
    # empate otra vez, el título más temprano. Es el mismo criterio que usa
    # `fase6_rag._elegir_precepto` para la otra puerta: conviene que las dos
    # decidan igual, o el mismo artículo sale de dos maneras según quién lo
    # pida.
    def _arranque(g):
        _o = sorted(g, key=lambda z: int(z.get("chunk_index") or 0))
        _t = " ".join(str(_o[0].get("texto") or "").split())
        _t = re.sub(r"^\s*\[[^\]]{0,400}\]\s*", "", _t)
        return _t[:220].lower()

    _racimos: dict = {}
    for g in grupos.values():
        _racimos.setdefault(_arranque(g), []).append(g)
    _mejor = max(_racimos.values(),
                 key=lambda r: (len(r), max(len(g) for g in r),
                                -min(_orden_titulo(g[0]) for g in r)))
    elegidos = max(_mejor, key=lambda g: (len(g), -_orden_titulo(g[0])))
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
        p.append("  CÍTALOS POR SU CASO Y SU PÁRRAFO, tal como van escritos")
        p.append("  aquí. El cuadernillo es dónde está recogido, no la fuente:")
        p.append("  la fuente es el caso y lo que en él se resolvió.")
        for x in m.coidh:
            p.append(f"\n  {x.cita()} — {x.tema}")
            p.append(f"  {x.texto[:900]}")
    return "\n".join(p)


def bloque_para_razonar(m: Marco, tope: int = 7000) -> str:
    """El parámetro del asunto para PROPONER y RAZONAR, sin reglas de redacción.

    `bloque` es para el redactor: dice cuántas palabras lleva el marco y dónde
    va. Quien decide la calificación no escribe marco, lo USA: necesita el
    texto de los preceptos y nada más. Hasta el 24-sep-2026 la propuesta y la
    razón decidían sin ver ninguno.
    """
    if m is None or m.vacio():
        return ""
    p = ["", "═" * 71,
         "EL PARÁMETRO DE ESTE ASUNTO — los preceptos que el acervo encontró",
         "═" * 71,
         "Son la premisa mayor: la ley local o secundaria que se aplica y el",
         "parámetro constitucional y convencional con el que se lee. Úsalos",
         "sólo si deciden algo; si no vienen al caso, no los nombres."]
    for x in m.locales:
        p.append(f"\n  {x.fuente} — Artículo {x.articulo}\n  «{x.texto[:1100]}»")
    for x in m.constitucionales:
        p.append(f"\n  Artículo {x.articulo} de la Constitución\n  {x.texto[:900]}")
    for x in m.convencionales:
        p.append(f"\n  {x.fuente} · {x.articulo}\n  {x.texto[:600]}")
    for x in m.coidh:
        p.append(f"\n  Corte Interamericana: {x.cita()} — {x.tema}\n  {x.texto[:500]}")
    t = "\n".join(p)
    return t[:tope]


# ═══════════════════════════════════════════════════════════════════════════
# LA FUNCIÓN QUE FALTABA
#
# Este módulo estaba entero pensado —el mapa temático, la consulta por
# conceptos y no por la prosa del caso, la reunión de artículos partidos, la
# trampa de los DOS artículos 17 de la Constitución— y nunca se escribió la que
# une todas esas piezas. `main.py` llamaba a `_mj.construir(...)`, que no
# existía, el AttributeError se tragaba en un `except Exception` y el marco
# jurídico salía SIEMPRE VACÍO. En los dos endpoints de resolver, desde el
# primer día, sin que nada avisara.
#
# Es el mismo patrón que ya costó meses con HyDE: una capa apagada se ve
# exactamente igual que una capa que funciona —el proyecto sale igual, sólo que
# peor— y por eso el aviso de abajo dice cuándo NO encontró nada, en vez de
# devolver un vacío mudo.
# ═══════════════════════════════════════════════════════════════════════════

MAX_FRAGMENTOS = 80


# ═══ LOS PRECEPTOS QUE CITA LA PROPIA RESPONSABLE ════════════════════════════
#
# David, 13-sep-2026: «el acto reclamado se rige por sus normas —en este caso el
# Código de Procedimientos Civiles— QUE GENERALMENTE CITA LA PROPIA RESPONSABLE.
# ¿Cómo podemos darle ese entendimiento al taller? Lo que debería citar son los
# artículos del código de procedimientos civiles».
#
# Es una señal mucho mejor que cualquier parecido de vectores: la autoridad que
# dictó el acto escribió con qué lo fundó. No hay que adivinarlo, hay que
# LEERLO. Y es determinista: o está escrito en el documento o no está.
#
# Se busca la fórmula con que se funda un acto —«con fundamento en los artículos
# 199, 202 y 208 del Código de Procedimientos Civiles del Estado»— y se sacan
# los números Y el ordenamiento. Después esos artículos se traen ENTEROS del
# acervo estatal, que es lo que permite transcribirlos entre comillas como el
# propio marco manda.
# EL SUFIJO «BIS/TER» DENTRO DEL GRUPO DE NÚMEROS SE COMÍA EL ESPACIO y con él
# la frontera de palabra siguiente: cada pieza casaba por separado y la unión no
# casaba nunca. Se parte en dos pasos —la lista de artículos por un lado, el
# ordenamiento por otro, buscado en la cola de la misma oración—, que además se
# lee mejor.
# LA FORMA UNIVERSAL DE CITAR: una lista de artículos y, detrás, el
# ordenamiento. Cubre «con fundamento en los artículos 199 y 202 del Código X» y
# también «…99, fracción III, 200, 201 … así como 204 del Código X», que es como
# aparece de verdad en las recurridas. Entre el último número y el ordenamiento
# caben incisos y fracciones, y por eso el hueco admite hasta 120 caracteres sin
# punto ni punto y coma — que son las dos cosas que cierran una cita.
_RX_CITA_DE_LEY = re.compile(
    r"(?P<form>(?:con\s+)?fundamento\s+en\s+(?:lo\s+dispuesto\s+por\s+)?)?"
    r"(?:los\s+|el\s+)?art[íi]culos?\s+"
    r"(?P<nums>\d{1,4}[^.;]{0,120}?)"
    r"\b(?:del|de\s+l[ao]s?|de|para\s+el)\s+"
    r"(?P<ley>(?:C[óo]digo|Ley|Reglamento)[^.;,)]{3,80})", re.I)

_RX_LEY_DEL_JUICIO = re.compile(r"ley\s+de\s+amparo", re.I)

_RX_CABEZA_FUND = re.compile(
    r"(?:con\s+)?fundamento\s+en\s+(?:lo\s+dispuesto\s+por\s+)?"
    r"(?:el\s+|los\s+)?art[íi]culos?\s+"
    r"(?P<nums>\d{1,4}(?:\s*[oº°]?)?(?:\s*(?:,|y|e)\s*\d{1,4}(?:\s*[oº°]?)?){0,12})",
    re.I)
_RX_ORDENAMIENTO = re.compile(
    r"\b(?:del|de\s+l[ao]s?|de)\s+"
    r"((?:C[óo]digo|Ley|Reglamento)[^.;,)]{3,80})", re.I)

_RX_FUNDAMENTO = re.compile(
    r"(?:con\s+)?fundamento\s+en\s+(?:lo\s+dispuesto\s+por\s+)?"
    r"(?:el\s+|los\s+)?art[íi]culos?\s+"
    r"(?P<nums>\d{1,4}(?:\s*[oº°]?\s*(?:BIS|TER)?)?"
    r"(?:\s*(?:,|y|e)\s*\d{1,4}(?:\s*[oº°]?\s*(?:BIS|TER)?)?){0,12})"
    r"[^.;]{0,80}?del?\s+(?P<ley>(?:C[óo]digo|Ley|Reglamento)[^.;,)]{3,80})",
    re.I)


# EL NOMBRE DEL ESTADO, PARA RECONOCER SU LEY. Se saca de la colección estatal
# que ya viaja en el encargo: «leyes_queretaro» → «queretaro».
def _estado_de(coleccion: str) -> str:
    return re.sub(r"^leyes_", "", str(coleccion or "").strip().lower())


def _sin_acento(x: str) -> str:
    import unicodedata as _u
    return "".join(c for c in _u.normalize("NFD", str(x or ""))
                   if _u.category(c) != "Mn").lower()


def preceptos_de_la_responsable(texto: str, tope: int = 8,
                                coleccion_estatal: str = "") -> list:
    """[(artículo, ordenamiento)] con que se fundó el acto reclamado.

    LA FÓRMULA «CON FUNDAMENTO EN» NO BASTA, y lo enseñó la recurrida real del
    322/2025: sólo sale dos veces y las dos son del juez de distrito citando el
    artículo 124 de la Ley de Amparo. Los preceptos de la responsable están ahí,
    pero recitados sin fórmula:

        «…Código Civil del Estado; así como 99, fracción III, 200, 201, párrafo
         segundo inciso e), así como 204 del Código de Procedimientos Civiles
         para el Estado de Querétaro, 7, 8, fracción V, 27, 28 y 29…»

    Así que se lee la forma UNIVERSAL de citar —una lista de artículos seguida
    del ordenamiento— y se manda el ordenamiento MÁS CITADO. Un acto se funda en
    una ley; las menciones sueltas de otras son la cita de una tesis o una
    remisión, y quedan por debajo en el recuento.

    LA LEY DE AMPARO NO CUENTA: funda el juicio, no el acto. Ésa es la regla que
    David pidió que el sistema entendiera, y aquí es una línea.
    """
    if not (texto or "").strip():
        return []
    t = " ".join(texto.split())
    por_ley, donde = {}, {}
    for m in _RX_CITA_DE_LEY.finditer(t):
        ley = " ".join(m.group("ley").split()).strip(" ,.;")
        if _RX_LEY_DEL_JUICIO.search(ley):
            continue
        nums = re.findall(r"\d{1,4}", m.group("nums"))
        if not nums:
            continue
        por_ley.setdefault(ley, [])
        # Con fórmula expresa pesa el doble: es el acto fundándose, no una cita.
        peso = 2 if m.group("form") else 1
        donde[ley] = donde.get(ley, 0) + peso * len(nums)
        for n_ in nums:
            if n_ not in por_ley[ley]:
                por_ley[ley].append(n_)
    if not por_ley:
        return []

    # MANDA LA ENTIDAD, NO LA FRECUENCIA, y lo enseñó la recurrida real: el juez
    # de distrito cita el Código Federal de Procedimientos Civiles más veces que
    # ningún otro —los artículos 129 y 202, los de la documental pública— porque
    # es el supletorio con que él valora las pruebas. Pero la ley del ACTO es la
    # del estado donde se dictó: si la responsable es un juez de primera
    # instancia de Querétaro, su ley lleva «del Estado de Querétaro» en el
    # nombre. Esa es la señal, y es determinista.
    _edo = _sin_acento(_estado_de(coleccion_estatal)).replace("_", " ")
    if _edo:
        del_estado = [k for k in por_ley if _edo in _sin_acento(k)]
        if del_estado:
            ley = max(del_estado, key=lambda k: donde.get(k, 0))
            return [(a, ley) for a in por_ley[ley][:tope]]
    ley = max(donde, key=lambda k: donde[k])
    return [(a, ley) for a in por_ley[ley][:tope]]


async def construir(qdrant, embed, problemas: list[str],
                    coleccion_estatal: Optional[str] = None,
                    texto_del_acto: str = "", puerta: bool = False,
                    temas_extra: Optional[list[str]] = None) -> Marco:
    """El bloque de constitucionalidad que ESTE asunto toca.

    Se busca con los CONCEPTOS de los artículos que el mapa temático disparó,
    no con el relato del expediente: está medido en este sistema que la prosa
    del caso no casa con la doctrina —el Cuadernillo No. 5 de la CoIDH tiene 424
    fragmentos y no salía ni uno buscándolo con «¿la pensión alimenticia del
    quince por ciento…?»—.
    """
    m = Marco()
    # EL TEMA TAMBIÉN VIVE EN LO QUE SE RESOLVIÓ Y EN LO QUE SE COMBATE, no sólo
    # en la pregunta (24-sep-2026). La pregunta del 711/2025 —«¿el bloqueo de
    # las cuentas de la persona moral podía extenderse…?»— no nombra ningún
    # derecho; lo nombra lo que resolvió el juzgado. Medido sobre 151 asuntos:
    # leyendo sólo las preguntas el mapa no disparaba NADA en 67 (44%); con
    # `resolvio` y `combate`, en 24. Sólo decide qué artículos entran: la
    # búsqueda se sigue haciendo con los conceptos y con las preguntas.
    _temas = list(problemas or []) + [t for t in (temas_extra or []) if t]
    arts = _articulos_del_problema(_temas)
    # LA PUERTA PROCESAL ABRE EL 17, Y DELANTE (24-sep-2026). Medido sobre 151
    # asuntos: 104 giran sobre una puerta procesal y en 84 el 17 no entraba,
    # porque el mapa temático lo enciende con «acceso a la justicia» y el
    # problema dice «extemporaneidad». La clase del problema lo sabe sin
    # palabras. Va DELANTE, con el 1º, porque el cupo es de tres y se llena en
    # el orden de esta lista: al final, el recorte se lo llevaba.
    if puerta:
        arts = ["1", "17"] + [a for a in arts if a not in ("1", "17")]
    if not arts:
        m.avisos.append(
            "El asunto no disparó ningún artículo del mapa constitucional: no "
            "se trajo bloque de constitucionalidad. Si el proyecto necesita "
            "uno, cítalo tú.")
        return m

    consulta = _consulta_convencional(arts, problemas)
    try:
        v = await embed(consulta)
    except Exception as ex:
        m.avisos.append(f"No se pudo vectorizar la consulta del marco: {ex}")
        return m

    from qdrant_client.http import models as _qm

    def _filtro(tipos: list[str], con_caso: bool = False):
        f = _qm.Filter(must=[_qm.FieldCondition(
            key="tipo", match=_qm.MatchAny(any=tipos))])
        if con_caso:
            # EL CASO SE EXIGE EN LA CONSULTA, no después. Descartarlo al leer
            # dejaba el marco sin precedente: medido en el 322/2025, los VEINTE
            # fragmentos más cercanos de la Corte venían sin caso identificado y
            # el aviso decía «no se citan» sobre una lista vacía. Pidiéndolo a
            # Qdrant, los que vuelven ya son citables.
            f.must_not = [_qm.IsEmptyCondition(
                is_empty=_qm.PayloadField(key="caso"))]
        return f

    consti, conven, coidh = await asyncio.gather(
        _buscar(qdrant, COLECCION, "dense", v, MAX_FRAGMENTOS,
                _filtro(["constitucion"])),
        _buscar(qdrant, COLECCION, "dense", v, MAX_FRAGMENTOS // 2,
                _filtro(["convencion"])),
        # ANCHO POR EL 61% QUE SE DESCARTA. Sólo 2,202 de los 5,518 fragmentos
        # de la Corte traen caso identificado; pidiendo seis para quedarse con
        # dos, el filtro dejaba el marco sin precedente la mitad de las veces.
        _buscar(qdrant, COLECCION, "dense", v, MAX_COIDH * 4,
                _filtro(["cuadernillo", "sentencia_cidh", "opinion_consultiva"],
                        con_caso=True)))

    # ── CONSTITUCIONALES ─────────────────────────────────────────────────
    # El mapa temático manda: sólo entran los artículos que ESTE asunto
    # disparó. Lo demás que la búsqueda traiga es ruido con buena puntuación,
    # que es la peor clase de error.
    por_art: dict = {}
    for p in consti:
        if _es_transitorio(p):
            continue
        a, _parte = _articulo_de(p)
        if a and a in arts:
            por_art.setdefault(a, []).append(p)
    for a in arts:
        frags = por_art.get(a) or []
        if not frags:
            m.avisos.append(
                f"El artículo {a} constitucional venía al caso por su tema y "
                f"NO se encontró en el acervo: no se cita.")
            continue
        m.constitucionales.append(Precepto(
            fuente="Constitución Política de los Estados Unidos Mexicanos",
            articulo=a, texto=_reunir_articulo(frags),
            jerarquia=str(frags[0].get("jerarquia") or ""),
            orden=_orden_titulo(frags[0])))
        if len(m.constitucionales) >= MAX_CONSTITUCIONALES:
            break
    m.constitucionales.sort(key=lambda x: (x.orden, int(x.articulo or 0)))

    # ── CONVENCIONALES, sólo si el asunto llama al bloque ─────────────────
    # La puerta procesal lo llama siempre: los artículos 8.1 y 25 de la
    # Convención son la otra mitad del parámetro del acceso a la justicia.
    if _pide_convencional(_temas) or puerta:
        vistos = set()
        for p in conven:
            ref = str(p.get("ref") or "").strip()
            fuente = str(p.get("origen") or "").strip() or ref
            if not ref or ref in vistos:
                continue
            vistos.add(ref)
            m.convencionales.append(Precepto(
                fuente=fuente, articulo=ref, texto=str(p.get("texto") or ""),
                jerarquia=str(p.get("jerarquia") or "")))
            if len(m.convencionales) >= MAX_CONVENCIONALES:
                break
        _sin_caso = 0
        for p in coidh:
            if len(m.coidh) >= MAX_COIDH:
                break
            caso = str(p.get("caso") or "").strip()
            if not caso:
                # SIN CASO NO HAY CITA. Antes se rellenaba con el `ref` del
                # fragmento y salía «el Caso CoIDH, Cuadernillo No. 2», que no
                # cita nada. Se descarta y se sigue mirando.
                _sin_caso += 1
                continue
            m.coidh.append(Precedente(
                caso=caso,
                vs=str(p.get("vs") or "").strip(),
                cuadernillo=str(p.get("cuadernillo_num")
                                or p.get("origen") or "").strip(),
                tema=str(p.get("cuadernillo_tema")
                         or p.get("tema_articulo") or "").strip(),
                parrafo=str(p.get("parrafo") or "").strip(),
                texto=str(p.get("texto") or ""),
                serie=str(p.get("serie_c") or "").strip()))
        if _sin_caso and not m.coidh:
            m.avisos.append(
                f"El acervo devolvió {_sin_caso} fragmentos de la Corte "
                f"Interamericana SIN caso identificado. No se citan: un "
                f"cuadernillo no es jurisprudencia, lo es el caso que contiene.")
        if not m.convencionales and not m.coidh:
            m.avisos.append(
                "El asunto llamaba a fuente convencional y el acervo no "
                "devolvió ninguna: no se cita ningún tratado.")

    # ═══ LA LEY DEL ACTO, QUE ES LA QUE LA RESPONSABLE APLICÓ ════════════════
    #
    # David: «el acto reclamado se rige por sus normas —el Código de
    # Procedimientos Civiles— que GENERALMENTE CITA LA PROPIA RESPONSABLE. ¿Cómo
    # podemos darle ese entendimiento al taller?».
    #
    # Así: la responsable dice QUÉ CÓDIGO —eso se lee, no se adivina— y dentro de
    # ese código se buscan los artículos del punto. Determinista para la ley,
    # semántico para el precepto. `Marco.locales` estaba declarado desde el
    # principio y nunca se llenaba: éste es su contenido.
    if coleccion_estatal and texto_del_acto:
        citados = preceptos_de_la_responsable(
            texto_del_acto, coleccion_estatal=coleccion_estatal)
        # UNA LEY FEDERAL NO SE BUSCA EN EL ACERVO DE UN ESTADO. Cuando la
        # responsable sólo cita leyes federales —una Sala del Tribunal FEDERAL
        # de Justicia Administrativa con sede en Querétaro—, el lector de arriba
        # no encuentra ninguna «del Estado» y devuelve la más citada: la Ley
        # Federal de Procedimiento Contencioso Administrativo. Buscarla dentro
        # de `leyes_queretaro` sólo puede devolver OTRA ley, y devolvió su
        # espejo estatal: la revisión fiscal 2/2026 salió con «En el ámbito
        # local, el artículo 57 de la Ley de Procedimiento Contencioso
        # Administrativo del Estado de Querétaro», copia literal del 51 federal.
        # La ley del acto es federal: el marco no lleva ley local, y lo federal
        # entra por su propio camino.
        _acto_federal = False
        try:
            import litis_normativa as _ln_m
            if citados and not _ln_m.es_local(citados[0][1]):
                print(f"   ⚖️ la ley del acto es federal ({citados[0][1][:60]}): "
                      f"no se busca en {coleccion_estatal}")
                citados, _acto_federal = [], True
        except Exception:
            pass
        if citados:
            _ley = citados[0][1]
            # NO SE FILTRA POR EL NOMBRE EXACTO, y costó descubrirlo: la
            # recurrida escribe «Código de Procedimientos Civiles PARA el Estado
            # de Querétaro» y el acervo lo tiene como «DEL Estado de Querétaro».
            # Un MatchValue con el nombre citado devuelve CERO. El nombre sirve
            # para PREFERIR, no para filtrar.
            # CON EL HECHO, NO CON LA CONSULTA CONVENCIONAL. `v` se armó con
            # los CONCEPTOS de los artículos constitucionales que el mapa
            # temático disparó —derechos humanos— y con ese vector el acervo
            # estatal devuelve leyes de derechos: la de Justicia para
            # Adolescentes, la de servicios. Medido: con el vector del HECHO
            # —la medida de restricción, el domicilio, la violencia familiar—
            # devuelve el artículo 202 del Código de Procedimientos Civiles,
            # «Medidas judiciales de protección en violencia familiar», y el
            # 256, «providencias precautorias en caso de violencia contra las
            # mujeres». Es la misma lección que la cesta del acto en fase6_rag.
            try:
                v_local = await embed(" ".join(
                    " ".join(str(x or "").split()) for x in problemas)[:600])
            except Exception:
                v_local = v
            # SE PESCA ANCHO Y SE QUEDA LO DE SU CÓDIGO. Con doce resultados
            # sólo uno era del código citado y los otros dos del marco salían de
            # leyes vecinas; el acervo estatal tiene muchas leyes y la del acto
            # compite con todas. Pescando treinta y dos y quedándose con las del
            # ordenamiento que la responsable nombró, el marco se llena con su
            # ley y no con la de al lado.
            locales = await _buscar(qdrant, coleccion_estatal, "dense", v_local,
                                    MAX_LOCALES * 11)
            # SE COMPARA POR LO QUE DISTINGUE, NO POR LO QUE COMPARTEN TODAS.
            # «Código de Procedimientos Civiles para el Estado de Querétaro» y
            # «Ley de Justicia para Adolescentes del Estado de Querétaro»
            # comparten estado, querétaro y para: con esas palabras dentro, el
            # parecido daba 3 para las dos y el marco se llenó de la ley de
            # adolescentes. Quitando las genéricas quedan «procedimientos» y
            # «civiles», que es lo que de verdad la nombra.
            _GENERICAS = {"estado", "queretaro", "querétaro", "para", "del",
                          "los", "las", "codigo", "código", "ley", "leyes",
                          "libre", "soberano", "republica", "república"}
            _clave = {w for w in re.findall(r"[a-záéíóúñ]{4,}", _ley.lower())
                      if w not in _GENERICAS}

            def _parecido(pl: dict) -> int:
                nom = str(pl.get("cuerpo_legal_oficial") or pl.get("ley")
                          or pl.get("origen") or "").lower()
                return len(_clave & {w for w in re.findall(r"[a-záéíóúñ]{4,}", nom)
                                     if w not in _GENERICAS})

            # Dos palabras distintivas coincidiendo ya es el mismo cuerpo legal:
            # «procedimientos» y «civiles» no las comparte ninguna otra ley del
            # acervo estatal.
            _minimo = min(2, len(_clave)) or 1
            # Y EL FUERO NO SE CRUZA aunque dos palabras coincidan. El parecido
            # por palabras distintivas es bueno para preferir un código dentro
            # del estado, pero «procedimiento contencioso administrativo» lo
            # comparten la ley federal y la de cada entidad. `misma_ley` es la
            # que sabe que «Federal» nunca es «del Estado».
            try:
                import fase6_rag as _f6r_m
                _misma_m = _f6r_m.misma_ley
            except Exception:
                _misma_m = None
            _suyas = [pl for pl in locales if _parecido(pl) >= _minimo
                      and (_misma_m is None or _misma_m(
                          _ley, str(pl.get("cuerpo_legal_oficial") or pl.get("ley")
                                    or pl.get("origen") or ""))
                           or _misma_m(str(pl.get("cuerpo_legal_oficial") or pl.get("ley")
                                           or pl.get("origen") or ""), _ley))]
            if _suyas:
                locales = _suyas
            else:
                locales.sort(key=lambda pl: -_parecido(pl))
                m.avisos.append(
                    f"EL MARCO VA SIN EL PRECEPTO DE «{_ley[:60]}»: el acervo "
                    f"estatal no devolvió ningún artículo de ese ordenamiento "
                    f"para este punto, aunque es el que la responsable citó. "
                    f"Compruébalo: el precepto que se transcribe es el que "
                    f"funda el acto.")
            vistos_l = set()
            for p in locales:
                art = str(p.get("articulo_num") or "").strip()
                cuerpo = str(p.get("cuerpo_legal_oficial") or p.get("ley")
                             or p.get("origen") or "").strip()
                if not art or (cuerpo, art) in vistos_l:
                    continue
                vistos_l.add((cuerpo, art))
                # LA CITA CON SU FUENTE. `ref` trae «Art. 202» y `capitulo` el
                # sitio del código donde vive: con los dos, el marco puede
                # transcribir el precepto diciendo de dónde sale, que es lo que
                # David echó en falta en los convencionales.
                m.locales.append(Precepto(
                    fuente=cuerpo or _ley,
                    articulo=str(p.get("ref") or f"Art. {art}").strip(),
                    texto=str(p.get("texto") or ""),
                    jerarquia=str(p.get("capitulo") or p.get("titulo") or "")))
                if len(m.locales) >= MAX_LOCALES:
                    break
            if m.locales:
                print(f"   ⚖️ ley del acto: {_ley[:60]} · "
                      f"{len(m.locales)} preceptos al marco "
                      f"(la responsable citó {', '.join(a for a, _ in citados[:6])})")
        elif not _acto_federal:
            # Sólo cuando de verdad no se leyó. Si se leyó y es federal, decir
            # «no se pudo leer» sería un aviso falso: el acto se lee bien, lo
            # que pasa es que no tiene ley local que transcribir.
            m.avisos.append(
                "NO SE PUDO LEER CON QUÉ LEY SE DICTÓ EL ACTO RECLAMADO: el "
                "documento no cita artículos de ningún código o ley del estado. "
                "El marco va sin el precepto local, que es el que se transcribe: "
                "compruébalo antes de firmar.")

    if m.vacio():
        m.avisos.append(
            "No se pudo construir el marco jurídico: el acervo no devolvió "
            "ningún precepto para los artículos del asunto.")
    return m
