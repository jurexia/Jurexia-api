"""Ensambla el adelanto sobre la plantilla REAL del secretario.

NO SE CONSTRUYE UN WORD NUEVO. Se abre el `.docx` del propio tribunal y se
rellenan sus huecos. La diferencia no es estética: en ese archivo viven la
cabecera, los estilos, los márgenes, la página de síntesis y las dos tablas de
calendario. Un documento generado desde cero se parece; éste ES el suyo.

La estructura, leída del ADA 240/2026 y confirmada por David:

    AMPARO DIRECTO ADMINISTRATIVO: 240/2026     ← número del asunto
    QUEJOSO: …
    MAGISTRADO: …                                ← ponente
    SECRETARIO DE TRIBUNAL: …
    V I S T O, para resolver …
    R E S U L T A N D O:
      PRIMERO. Presentación de la demanda        ┐
      SEGUNDO. Derechos humanos vulnerados       │
      TERCERO. Parte tercera interesada          │ machote + datos de la ficha
      CUARTO. Trámite del juicio                 │
      QUINTO. Turno del asunto                   │
      SEXTO. Verificación de la sesión remota    ┘
    C O N S I D E R A N D O:
      PRIMERO. Competencia                       ← machote por tipo de asunto
      SEGUNDO. Existencia del acto reclamado     ← machote
      TERCERO. Legitimación y oportunidad        ← FASE 0: el cómputo
      CUARTO. Sentencia reclamada y conceptos    ← «Es innecesario transcribir…»
      QUINTO. Antecedentes                       ← de la sentencia reclamada
      SEXTO. Estudio                             ← el corazón:
          · resumen de la sentencia reclamada        438 palabras, PASADO
          · resumen de los conceptos o agravios      472 palabras, PRESENTE
          · problemas jurídicos (o el global)
          · … y aquí entra el criterio del secretario
    Por lo expuesto y fundado, se resuelve:
    ÚNICO. La Justicia de la Unión ***** a …     ← el sentido lo pone él
    (página de síntesis: TEMA, OPORTUNIDAD y los dos calendarios)

CÓMO SE RELLENA SIN ROMPER EL FORMATO: cada párrafo de Word es una lista de
«runs», y el formato vive en los runs, no en el párrafo. Si se borra el párrafo
y se escribe otro, se pierde la fuente, el interlineado y la sangría. Aquí se
conserva SIEMPRE el primer run —que lleva el formato bueno—, se le cambia el
texto y se eliminan los demás. Para añadir párrafos se CLONA uno vecino, por lo
mismo.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

try:
    import docx
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.text.paragraph import Paragraph
    from docx.text.run import Run
except ImportError:                                     # pragma: no cover
    docx = None


# ═══════════════════════════════════════════════════════════════════════════
# Manipulación cuidadosa del .docx
# ═══════════════════════════════════════════════════════════════════════════

def texto_de(p) -> str:
    return "".join(n.text or "" for n in p._p.iter(qn("w:t")))


def escribir(p, texto: str) -> None:
    """Cambia el texto conservando el formato del párrafo.

    Se conserva el PRIMER run y se borran los demás: el formato —fuente,
    tamaño, negrita, sangría— vive en el run, no en el párrafo. Vaciar el
    párrafo y escribir de nuevo lo dejaría en el estilo por defecto.
    """
    runs = p.runs
    if not runs:
        p.add_run(texto)
        return
    runs[0].text = texto
    for r in runs[1:]:
        r._element.getparent().remove(r._element)


def escribir_tramos(p, tramos) -> None:
    """El párrafo con VARIOS formatos dentro: [(texto, {"bold":…, "italic":…}), …].

    `escribir()` mete todo en el primer run y hereda su formato, que es por qué
    el resolutivo salía ENTERO en negrita: el primer run de la plantilla es
    «ÚNICO.» y va en negrita. Aquí el primer run se usa como MOLDE —de él salen
    la fuente, el tamaño y el color— y se clona uno por tramo.
    """
    runs = p.runs
    if not runs:
        for texto, fmt in tramos:
            r = p.add_run(texto)
            r.bold = fmt.get("bold", False)
            r.italic = fmt.get("italic", False)
        return

    molde = runs[0]._element
    nuevos = []
    for texto, fmt in tramos:
        e = copy.deepcopy(molde)
        nuevos.append(e)
        molde.addprevious(e)
        r = Run(e, p)
        r.text = texto
        r.bold = fmt.get("bold", False)
        r.italic = fmt.get("italic", False)
    for r in list(p.runs):
        if r._element not in nuevos:
            r._element.getparent().remove(r._element)


# El rubro de una tesis se escribe entre comillas y EN NEGRITA; así lo hace él
# en los 5 casos del ADC 174-2026 y en todos los engroses revisados. La cita se
# reconoce por las comillas tipográficas y por ir en versales, que es como la
# publica la Corte.
_RX_RUBRO = re.compile(r"[“\"]([A-ZÁÉÍÓÚÑ][^”\"]{18,}?)[”\"]")


def tramos_con_rubro(texto: str):
    """Parte el párrafo en tramos para que el rubro salga en negrita."""
    tramos, i = [], 0
    for m in _RX_RUBRO.finditer(texto):
        if m.start() > i:
            tramos.append((texto[i:m.start()], {}))
        tramos.append((texto[m.start():m.end()], {"bold": True}))
        i = m.end()
    if not tramos:
        return None
    if i < len(texto):
        tramos.append((texto[i:], {}))
    return tramos


def clonar_tras(p, texto: str, modelo=None):
    """Un párrafo nuevo DESPUÉS de `p`, con el formato de `modelo`.

    EL MODELO NO ES UN DETALLE. Antes se clonaba siempre el párrafo ancla, y
    el ancla es el ENCABEZADO —«SEXTO. Estudio…»—, que va en negrita y con otra
    sangría. Resultado: el estudio entero salía en negrita y con el sangrado
    del título, y el secretario tenía que arreglar el formato a mano, que es
    justo el tiempo que este programa pretende ahorrarle.

    Medido en la plantilla de David hay TRES formatos distintos:
        encabezado  negrita, sangría 540385
        rótulo      negrita, SIN sangría
        cuerpo      sin negrita, sangría 457200
    """
    fuente = modelo if modelo is not None else p
    nuevo = copy.deepcopy(fuente._p)
    p._p.addnext(nuevo)
    q = Paragraph(nuevo, p._parent)
    escribir(q, texto)
    return q


# ═══════════════════════════════════════════════════════════════════════════
# Notas al pie DE VERDAD
# ═══════════════════════════════════════════════════════════════════════════
#
# Las citas a página NO van entre paréntesis en el cuerpo. David lo pidió así y
# es la convención del oficio: al pie, «Cfr. página 7, párrafo 3».
#
# `docx_generator_tcc.py` las finge con un superíndice y una lista al final del
# documento. Aquí se hacen de verdad, escribiendo en `word/footnotes.xml`, que
# la plantilla ya tiene —trae tres notas propias—, y copiando el formato de las
# suyas para que las nuevas no desentonen.

_NS_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _parte_notas(doc):
    """La parte `footnotes.xml`, o None si la plantilla no tiene notas.

    OJO: python-docx la entrega como `Part` GENÉRICO, no como parte XML — no
    tiene `.element`, sólo `.blob` con los bytes. Hay que analizarla y volver a
    escribirla a mano.
    """
    for rel in doc.part.rels.values():
        if rel.reltype.endswith("/footnotes"):
            return rel.target_part
    return None


def _leer_notas(parte):
    from lxml import etree
    return etree.fromstring(parte.blob)


def _guardar_notas(parte, raiz) -> None:
    from lxml import etree
    parte._blob = etree.tostring(raiz, xml_declaration=True,
                                 encoding="UTF-8", standalone=True)


def _siguiente_id(raiz) -> int:
    ids = [int(n.get(qn("w:id"))) for n in raiz.findall(qn("w:footnote"))
           if n.get(qn("w:id")) is not None]
    return max(ids) + 1 if ids else 1


def _estilo_referencia(doc) -> str:
    """El styleId que la plantilla usa para la llamada de nota al pie."""
    try:
        for st in doc.styles:
            if (st.name or "").strip().lower() in ("footnote reference",
                                                   "ref. de nota al pie",
                                                   "referencia de nota al pie"):
                return st.style_id
    except Exception:
        pass
    return "FootnoteReference"


def anadir_nota(doc, parrafo, texto: str) -> bool:
    """Cuelga una nota al pie del final de `parrafo`. Devuelve si pudo.

    Se clona una nota EXISTENTE de la plantilla para heredar su estilo —tamaño,
    fuente, sangría— en vez de fabricar una desde cero, que saldría con el
    formato por defecto y cantaría al lado de las suyas.
    """
    parte = _parte_notas(doc)
    if parte is None:
        return False
    raiz = _leer_notas(parte)
    modelo = None
    for n in raiz.findall(qn("w:footnote")):
        tipo = n.get(qn("w:type"))
        if tipo in (None, "normal") and n.findall(qn("w:p")):
            modelo = n
            break
    if modelo is None:
        return False

    nid = _siguiente_id(raiz)
    nueva = copy.deepcopy(modelo)
    nueva.set(qn("w:id"), str(nid))
    # Se deja UN párrafo y se le escribe el texto MANIPULANDO EL XML, no con
    # `Paragraph`: la parte se analiza con lxml plano y sus elementos no son
    # los `CT_P` de python-docx, así que `.runs` no existe ahí.
    ps = nueva.findall(qn("w:p"))
    for extra in ps[1:]:
        nueva.remove(extra)
    textos = ps[0].findall(f".//{qn('w:t')}") if ps else []
    if textos:
        textos[0].text = texto
        textos[0].set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        for sobra in textos[1:]:
            sobra.text = ""
    raiz.append(nueva)
    _guardar_notas(parte, raiz)

    # La llamada en el cuerpo: un run con estilo de referencia Y con superíndice
    # explícito. El estilo por nombre —«Refdenotaalpie»— sólo se aplica si la
    # plantilla lo define; cuando no existe, Word ignora la referencia y el
    # número sale como texto corrido pegado a la palabra, que es lo que David vio
    # como «caracteres incorrectos en lugar del número». El superíndice directo
    # no depende de ningún estilo.
    r = parrafo.add_run()
    rpr = r._element.get_or_add_rPr()
    # El styleId REAL de la plantilla, leído de ella: «FootnoteReference». El
    # nombre que yo puse —«Refdenotaalpie»— es el rótulo español de Word y NO
    # existe como identificador ahí, así que Word lo ignoraba en silencio.
    estilo = OxmlElement("w:rStyle")
    estilo.set(qn("w:val"), _estilo_referencia(doc))
    rpr.append(estilo)
    va = OxmlElement("w:vertAlign")
    va.set(qn("w:val"), "superscript")
    rpr.append(va)
    ref = OxmlElement("w:footnoteReference")
    ref.set(qn("w:id"), str(nid))
    r._element.append(ref)
    return True


# ── El formato de la TRANSCRIPCIÓN de una tesis ──────────────────────────
#
# Medido sobre 1,676 transcripciones reales del corpus, y es DISTINTO del
# cuerpo en cuatro cosas a la vez:
#
#                    cuerpo (17,715 párr.)   transcripción (1,676)
#     sangría izq.   ninguna (91%)           709 twips (48%)  ← 1.25 cm
#     1ª línea       709 twips (55%)         ninguna (80%)
#     interlineado   1.5 (75%)               1.0 (48%)
#     tamaño         14 pt (76%)             13 pt (72%)
#     cursiva        no (97%)                SÍ (100%)
#
# Clonarla del cuerpo —que es lo que se hacía— la deja sin sangrar, a 14 puntos
# y con interlineado y medio: se lee como si fuera prosa del tribunal y no como
# lo que es, texto ajeno transcrito.
SANGRIA_CITA = 709          # twips = 1.25 cm

# La sangría de PRIMERA LÍNEA del cuerpo. El corpus la usa en el 63-70% de los
# párrafos (709 twips), pero David la ve «muy desplazada» y su propio proyecto
# terminado del ADC 174-2026 va a CERO. Su criterio manda sobre la moda del
# corpus: son sus sentencias y él es quien las firma.
SANGRIA_PARRAFO = 0
TAMANO_CITA = 13            # puntos


def _sangrar(p) -> None:
    """La sangría del cuerpo, uniforme en todo lo que generamos."""
    from docx.shared import Twips
    p.paragraph_format.first_line_indent = Twips(SANGRIA_PARRAFO)


def _aplicar_formato_cita(p) -> None:
    """Sangra, achica e inclina un párrafo ya escrito."""
    from docx.shared import Pt, Twips
    pf = p.paragraph_format
    pf.left_indent = Twips(SANGRIA_CITA)
    pf.first_line_indent = Twips(0)
    pf.line_spacing = 1.0
    for r in p.runs:
        r.italic = True
        r.font.size = Pt(TAMANO_CITA)


def formato_cita(modelo_cuerpo):
    """El modelo del cuerpo, ajustado a las medidas de la transcripción."""
    if modelo_cuerpo is None:
        return None
    from docx.shared import Pt, Twips
    nuevo = copy.deepcopy(modelo_cuerpo._p)
    q = Paragraph(nuevo, modelo_cuerpo._parent)
    pf = q.paragraph_format
    pf.left_indent = Twips(SANGRIA_CITA)
    pf.first_line_indent = Twips(0)
    pf.line_spacing = 1.0
    for r in q.runs:
        r.italic = True
        r.font.size = Pt(TAMANO_CITA)
    return q


def modelos_de_formato(doc) -> dict:
    """Un párrafo de muestra por cada formato, tomado de la propia plantilla.

    HAY QUE LLAMARLA ANTES DE BORRAR NADA: los modelos son los párrafos que el
    secretario ya escribió, y al limpiar la sección para rellenarla se van con
    ella.
    """
    rotulo = cuerpo = None
    for p in doc.paragraphs:
        t = texto_de(p).strip()
        if not t or not p.runs:
            continue
        negrita = bool(p.runs[0].bold)
        palabras = len(t.split())
        # Rótulo: negrita, corto y sin numeral ordinal delante.
        if rotulo is None and negrita and 2 <= palabras <= 9 \
                and not re.match(r"^(PRIMERO|SEGUNDO|TERCERO|CUARTO|QUINTO|SEXTO|S[ÉE]PTIMO)\.", t):
            rotulo = p
        # Cuerpo: prosa larga sin negrita.
        if cuerpo is None and not negrita and palabras > 25:
            cuerpo = p
        if rotulo is not None and cuerpo is not None:
            break
    # El párrafo VACÍO también es un modelo, y es el que faltaba.
    #
    # David lo dijo mirando el documento: «sin espacios». No era una propiedad
    # de espaciado: en su plantilla hay UN PÁRRAFO VACÍO entre cada párrafo de
    # texto —la secuencia real es T · T · T ·— y yo los escribía pegados.
    vacio = next((p for p in doc.paragraphs if not texto_de(p).strip()), None)
    return {"rotulo": rotulo, "cuerpo": cuerpo, "vacio": vacio}


# Las marcas que deja el pipeline: [[p.7 §3]] o [[p.7]].
# El modelo no siempre ancla un párrafo suelto: cuando la idea abarca varios
# escribe «[[p.39 §2-3]]», y a veces encadena dos marcas para dos páginas. El
# patrón original sólo admitía «§3» y esas marcas se quedaban CRUDAS dentro de
# la sentencia — visibles, con corchetes, en el documento que se firma.
# Y no siempre es UNA cita: cuando la idea abarca dos páginas el modelo escribe
# «[[p.33 §1; p.34 §§1-2]]» —dos referencias dentro de la misma marca, con «§§»
# para el plural—. El patrón anterior exigía una sola y esas marcas se quedaban
# CRUDAS, con corchetes, justo donde debía ir el número de la nota al pie. Ahora
# se captura la marca ENTERA y se despieza aparte.
_MARCA_CITA = re.compile(r"\s*\[\[([^\[\]]{2,120})\]\]")
_UNA_CITA = re.compile(
    r"p{1,2}\.?\s*(\d{1,4})(?:\s*[-–]\s*\d{1,4})?"
    r"(?:\s*§+\s*(\d{1,3}(?:\s*[-–]\s*\d{1,3})?))?", re.I)


def _citas_de(marca: str) -> list[tuple[str, str]]:
    """Las (página, párrafo) que hay dentro de una marca, sean una o varias."""
    fuera = []
    for trozo in re.split(r"[;,]", marca):
        m = _UNA_CITA.search(trozo)
        if m:
            fuera.append((m.group(1), m.group(2) or ""))
    return fuera


def _texto_de_nota(pagina: str, parrafo: Optional[str]) -> str:
    if not parrafo:
        return f"Cfr. página {pagina}."
    p = re.sub(r"\s*[-–]\s*", " a ", parrafo)
    return (f"Cfr. página {pagina}, párrafos {p}." if " a " in p
            else f"Cfr. página {pagina}, párrafo {p}.")


def _normaliza_rubro(x: str) -> str:
    return re.sub(r"[^A-ZÁÉÍÓÚÑ0-9]+", " ", (x or "").upper()).strip()


def tesis_del_rubro(texto: str, tesis: list) -> Optional[dict]:
    """La tesis del acervo cuyo rubro se cita en este párrafo, si alguna.

    Se compara NORMALIZANDO —fuera comillas, acentos de puntuación y espacios—
    porque el modelo reproduce el rubro con comillas tipográficas y a veces
    corta la coletilla final.
    """
    m = _RX_RUBRO.search(texto or "")
    if not m:
        return None
    citado = _normaliza_rubro(m.group(1))
    if len(citado) < 25:
        return None
    for t in (tesis or []):
        real = _normaliza_rubro(t.get("rubro", ""))
        if not real:
            continue
        if real.startswith(citado[:70]) or citado.startswith(real[:70]):
            return t
    return None


def clonar_bloque(ancla, textos, modelo, modelo_vacio, doc=None, tesis=None,
                  modelo_cita=None):
    """Varios párrafos con su separador vacío detrás, como los escribe él.

    Y con las marcas de cita convertidas en NOTAS AL PIE. Si el documento no
    admite notas, la marca se retira del texto: nunca se deja un «[[p.7 §3]]»
    a la vista.
    """
    for t in textos:
        citas = [c for m in _MARCA_CITA.finditer(t) for c in _citas_de(m.group(1))]
        limpio = _MARCA_CITA.sub("", t)
        ancla = clonar_tras(ancla, limpio, modelo)
        _sangrar(ancla)

        # El rubro va en NEGRITA dentro del párrafo, como él lo escribe.
        tramos = tramos_con_rubro(limpio)
        if tramos:
            escribir_tramos(ancla, tramos)

        if doc is not None:
            for pagina, parrafo in citas:
                anadir_nota(doc, ancla, _texto_de_nota(pagina, parrafo))

        # Y la tesis se completa desde el acervo: la LOCALIZACIÓN a pie de
        # página —«Semanario Judicial…, Novena Época, tomo XXXI, página 830»— y
        # el TEXTO ÍNTEGRO en párrafo aparte y en cursiva. Citar sólo el rubro
        # deja al lector sin poder comprobar qué dice la tesis.
        hallada = tesis_del_rubro(limpio, tesis) if tesis else None
        if hallada:
            if doc is not None and hallada.get("localizacion"):
                anadir_nota(doc, ancla, " " + hallada["localizacion"].strip())
            cuerpo = (hallada.get("texto") or "").strip()
            if cuerpo:
                # El rubro y su transcripción son UNA cita partida en dos
                # párrafos. Sin esto Word los separa en el salto de página y la
                # tesis aparece dos hojas después de lo que anuncia.
                ancla.paragraph_format.keep_with_next = True
                if modelo_vacio is not None:
                    ancla = clonar_tras(ancla, "", modelo_vacio)
                ancla.paragraph_format.keep_with_next = True   # el vacío también
                ancla = clonar_tras(ancla, cuerpo, modelo_cita or modelo)
                escribir_tramos(ancla, [(cuerpo, {"italic": True})])
                # El formato va DESPUÉS de escribir: `escribir_tramos` clona el
                # primer run como molde y heredaría el tamaño del cuerpo.
                _aplicar_formato_cita(ancla)

        if modelo_vacio is not None:
            ancla = clonar_tras(ancla, "", modelo_vacio)
    return ancla


def buscar(doc, patron: str, desde: int = 0):
    """El primer párrafo, a partir del índice `desde`, que empieza por `patron`.

    EL `desde` NO ES UN LUJO. Los ordinales se repiten: hay un «SEXTO.
    Verificación de la sesión» en el RESULTANDO y un «SEXTO. Estudio» en el
    CONSIDERANDO, y el primero va ANTES que el «QUINTO. Antecedentes». Buscar
    «^SEXTO\.» a secas devuelve el equivocado y deja un rango invertido que no
    borra nada — pasó, y los antecedentes viejos se quedaron en el documento.
    """
    rx = re.compile(patron, re.I)
    for i, p in enumerate(doc.paragraphs):
        if i >= desde and rx.match(texto_de(p).strip()):
            return p
    return None


def indice_de(doc, p) -> int:
    for i, q in enumerate(doc.paragraphs):
        if q._p is p._p:
            return i
    return 0


def borrar_entre(doc, ini, fin) -> int:
    """Vacía los párrafos ENTRE dos marcas ya localizadas, sin tocarlas.

    Recibe los párrafos, no patrones: quien llama ya se aseguró de que `fin`
    va DESPUÉS de `ini`. Con patrones sueltos el rango puede salir invertido y
    entonces no borra nada en silencio, que es peor que fallar.
    """
    if ini is None or fin is None:
        return 0
    cuerpo = ini._p.getparent()
    hijos = list(cuerpo.iterchildren())
    try:
        a, b = hijos.index(ini._p), hijos.index(fin._p)
    except ValueError:
        return 0
    n = 0
    for el in hijos[a + 1:b]:
        if el.tag == qn("w:p"):
            cuerpo.remove(el)
            n += 1
    return n


# ═══════════════════════════════════════════════════════════════════════════
# Lo que se rellena
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Relleno:
    """Todo lo que el ensamblador necesita saber del asunto."""
    # Cabecera
    encabezado: str = ""                 # «AMPARO DIRECTO ADMINISTRATIVO: 240/2026»
    quejoso: str = ""
    magistrado: str = ""
    secretario: str = ""

    # Fase 0 — el párrafo del cómputo, ya redactado
    oportunidad: str = ""

    # QUINTO. Antecedentes — de la sentencia reclamada
    antecedentes: list[str] = field(default_factory=list)

    # SEXTO. Estudio — los tres apartados, en este orden
    resumen_acto: list[str] = field(default_factory=list)
    resumen_conceptos: list[str] = field(default_factory=list)
    problemas: list[str] = field(default_factory=list)

    # Fase 6 — el estudio de fondo, ya con el criterio del secretario dentro.
    # Vacío mientras no lo haya dictado: entonces el documento es un ADELANTO,
    # con su hueco, y no una sentencia a medio resolver.
    estudio: list[str] = field(default_factory=list)

    # Las tesis del acervo (registro, rubro, texto, localizacion). El ensamblador
    # TRANSCRIBE de aquí, no de lo que escriba el modelo: la transcripción de una
    # tesis es dato verificado, y hacérsela redactar es invitarle a alterarla.
    tesis: list[dict] = field(default_factory=list)

    # Lo que decide el secretario. Mientras esté vacío se deja el hueco
    # marcado, igual que en el adelanto de papel.
    sentido: str = ""                    # la FÓRMULA literal, si ya se tiene
    calificaciones: list[str] = field(default_factory=list)   # o las del fondo
    tema: str = ""

    es_recurso: bool = False             # revisión → «agravios», no «conceptos»
    numero_asunto: str = ""              # «512/2026», para el VISTO y el resolutivo


HUECO = "*********"

# Lo que el ensamblador quiere decir sobre el documento que acaba de escribir.
avisos_ensamblado: list[str] = []

# ── La calificación de los conceptos NO ES el resolutivo ──────────────────
#
# «ineficaces» califica los conceptos de violación; el resolutivo dice si la
# Justicia de la Unión ampara o no. Meter la primera donde va el segundo produce
# «La Justicia de la Unión ineficaz a María Fernanda Ruiz», que no significa
# nada y que es exactamente la línea que se firma.
# En MINÚSCULAS y en negrita: así lo escribe él —«La Justicia de la Unión **no
# ampara ni protege** a…»—. Las versales son de otro tribunal, no del suyo.
_AMPARA = "ampara y protege"
_NO_AMPARA = "no ampara ni protege"


def formula_resolutivo(calificaciones: list[str]) -> tuple[str, str]:
    """(fórmula, aviso). Basta un concepto fundado para que se conceda.

    Con calificaciones mixtas la fórmula es la de concesión, pero el resolutivo
    real lleva además los EFECTOS, y esos no los escribe una tabla: se avisa
    para que los ponga quien firma.
    """
    cs = [str(c or "").strip().lower() for c in calificaciones if str(c or "").strip()]
    if not cs:
        return "", ""
    hay_fundado = any(c.startswith("fundad") for c in cs)
    if not hay_fundado:
        return _NO_AMPARA, ""
    if len(set(cs)) > 1:
        return _AMPARA, ("El resolutivo concede porque hay conceptos fundados, "
                         "pero la calificación es mixta: los EFECTOS de la "
                         "concesión los tienes que redactar tú.")
    return _AMPARA, ""


def _quejoso_de_plantilla(ruta: str) -> str:
    """El quejoso que trae la plantilla, para poder sustituirlo en todo el texto."""
    d = docx.Document(ruta)
    for p in d.paragraphs:
        t = texto_de(p).strip()
        m = re.match(r"^QUEJOS[OA]\s*:\s*(.+?)\.?$", t, re.I)
        if m:
            return m.group(1).strip()
    return ""


def ensamblar(ruta_plantilla: str, r: Relleno, ruta_salida: str) -> str:
    """Rellena la plantilla y guarda el adelanto. Devuelve la ruta escrita."""
    avisos_ensamblado.clear()
    doc = docx.Document(ruta_plantilla)
    q = "agravios" if r.es_recurso else "conceptos de violación"
    # PRIMERO los modelos de formato: luego se borra el contenido viejo y con
    # él se irían los ejemplos de los que se copia el estilo.
    mod = modelos_de_formato(doc)

    # ── Cabecera ─────────────────────────────────────────────────────────
    # Aparece DOS veces: al inicio y en la página de síntesis. Se cambian las
    # dos, que es justo lo que se olvida al hacerlo a mano.
    # LOS DOS PUNTOS SON OBLIGATORIOS EN EL PATRÓN. Sin ellos, «^MAGISTRAD»
    # casa con «Magistrado Instructor adscrito a la Segunda Sección…» —que es
    # la AUTORIDAD RESPONSABLE, dentro del resultando— y la sobrescribe. Pasó.
    for patron, valor in (
        (r"^AMPARO\s+[\w\s]+:", r.encabezado),
        (r"^QUEJOS[OA]\s*:", f"QUEJOSO: {r.quejoso}."),
        (r"^MAGISTRADO(?:\s+PONENTE)?\s*:", f"MAGISTRADO: {r.magistrado}."),
        (r"^SECRETARIO(?:\s+DE\s+TRIBUNAL)?\s*:", f"SECRETARIO DE TRIBUNAL: {r.secretario}."),
    ):
        if not valor:
            continue
        rx = re.compile(patron, re.I)
        for p in doc.paragraphs:
            if rx.match(texto_de(p).strip()):
                escribir(p, valor)

    # ── TERCERO. Legitimación y oportunidad ──────────────────────────────
    if r.oportunidad:
        p = buscar(doc, r"^TERCERO\.\s*Legitimaci")
        if p is not None:
            # El cómputo es el párrafo SIGUIENTE al de legitimación.
            sig = p._p.getnext()
            while sig is not None and sig.tag != qn("w:p"):
                sig = sig.getnext()
            if sig is not None:
                escribir(Paragraph(sig, p._parent), r.oportunidad)

    # ── QUINTO. Antecedentes ─────────────────────────────────────────────
    if r.antecedentes:
        p = buscar(doc, r"^QUINTO\.\s*Antecedentes")
        fin = buscar(doc, r"^SEXTO\.\s*Estudio", desde=indice_de(doc, p) + 1) if p else None
        borrar_entre(doc, p, fin)
        if p is not None:
            ancla = clonar_bloque(p, r.antecedentes, mod["cuerpo"], mod["vacio"], doc)

    # ── SEXTO. Estudio ───────────────────────────────────────────────────
    p = buscar(doc, r"^SEXTO\.\s*Estudio")
    if p is not None and (r.resumen_acto or r.resumen_conceptos or r.problemas
                          or r.estudio):
        fin = buscar(doc, r"^Por lo expuesto", desde=indice_de(doc, p) + 1)
        borrar_entre(doc, p, fin)
        escribir(p, f"SEXTO. Estudio de los {q}.")
        ancla = p
        bloques: list[tuple[str, list[str]]] = [
            ("Consideraciones relevantes de la sentencia recurrida"
             if r.estudio else
             "Consideraciones relevantes de la resolución reclamada.", r.resumen_acto),
            (q.capitalize() + ".", r.resumen_conceptos),
            # «Problema jurídico a resolver» y «Solución» son los rótulos del
            # ENGROSE; «Problemas jurídicos» era el del adelanto. Al llegar el
            # criterio, el documento deja de ser adelanto y toma los suyos.
            ("Problema jurídico a resolver" if r.estudio else "Problemas jurídicos.",
             r.problemas),
            # El estudio va SIN rótulo propio: abre con su encabezado ordinal y
            # la calificación —«Los conceptos son ineficaces»—, que es como
            # arranca el 40% de los engroses y lo que el lector busca primero.
            ("Solución", r.estudio),
        ]
        for rotulo, parrafos in bloques:
            if not parrafos:
                continue
            if rotulo:
                ancla = clonar_tras(ancla, rotulo, mod["rotulo"])
                if mod["vacio"] is not None:
                    ancla = clonar_tras(ancla, "", mod["vacio"])
            ancla = clonar_bloque(ancla, parrafos, mod["cuerpo"], mod["vacio"],
                                  doc, tesis=r.tesis)

    # ── El NOMBRE de la parte, allí donde la plantilla lo arrastra ───────
    #
    # El VISTO y el resolutivo traen el quejoso y el número del asunto ANTERIOR,
    # porque la plantilla es un adelanto real. Dejar ahí el nombre de otro es
    # más peligroso que dejar un hueco: un hueco se ve, un nombre equivocado
    # se firma. Se sustituye por el del asunto en curso.
    quejoso_viejo = _quejoso_de_plantilla(ruta_plantilla)

    # La plantilla puede traer el MISMO nombre escrito de dos maneras —en este
    # tribunal, «Larrañaga» en el proemio y «Larragaña» en el resolutivo—. Se
    # sustituye la forma canónica y la otra sobrevive intacta hasta la firma.
    # No se corrige automáticamente: un apellido no se arregla por parecido.
    if r.quejoso:
        _apellidos = [w for w in re.findall(r"[\wÁÉÍÓÚÑáéíóúñ]{5,}", r.quejoso)]
        for p_ in doc.paragraphs:
            t_ = texto_de(p_)
            for ap in _apellidos:
                for cand in re.findall(r"\b[\wÁÉÍÓÚÑáéíóúñ]{5,}\b", t_):
                    if cand.lower() == ap.lower():
                        continue
                    if (sorted(cand.lower()) == sorted(ap.lower())
                            and cand.lower() not in (x.lower() for x in _apellidos)):
                        avisos_ensamblado.append(
                            f"La plantilla escribe «{cand}» donde el quejoso es "
                            f"«{ap}». Compruébalo: se firma tal cual.")
                        break

    if r.quejoso and quejoso_viejo:
        for p in doc.paragraphs:
            t = texto_de(p)
            if quejoso_viejo.lower() in t.lower():
                escribir(p, re.sub(re.escape(quejoso_viejo), r.quejoso.title(),
                                   t, flags=re.I))
    if r.numero_asunto:
        rx_num = re.compile(r"\b\d{1,4}\s*/\s*\d{4}\b")
        for patron in (r"^V\s?I\s?S\s?T", r"^ÚNICO\.", r"^PRIMERO\.\s*(?:Se |La Justicia)"):
            q = buscar(doc, patron)
            if q is not None:
                escribir(q, rx_num.sub(r.numero_asunto, texto_de(q)))

    # ── El sentido y la síntesis: huecos si el secretario no ha decidido ──
    p = buscar(doc, r"^ÚNICO\.|^PRIMERO\.\s*(?:Se |La Justicia)")
    if p is not None:
        formula = r.sentido or formula_resolutivo(r.calificaciones)[0]
        # Sólo se toca el resolutivo de AMPARO. En un recurso la fórmula es otra
        # —se confirma, se revoca, se declara infundado— y depende de lo que se
        # recurrió: dejar el hueco es más honesto que rellenarlo por analogía.
        if formula and "justicia de la unión" in texto_de(p).lower():
            # El resolutivo NO va entero en negrita. Medido en su engrose: sólo
            # «ÚNICO.» y la fórmula. `escribir()` metía todo en el primer run,
            # que es el del ordinal, y heredaba su negrita: el resolutivo entero
            # salía resaltado y había que deshacerlo a mano.
            entero = texto_de(p).replace(HUECO, formula)
            m_ord = re.match(r"^(ÚNICO\.|PRIMERO\.|SEGUNDO\.)", entero)
            tramos = []
            resto = entero
            if m_ord:
                tramos.append((m_ord.group(1), {"bold": True}))
                resto = entero[m_ord.end():]
            i = resto.find(formula)
            if i >= 0:
                tramos += [(resto[:i], {}), (formula, {"bold": True}),
                           (resto[i + len(formula):], {})]
            else:
                tramos.append((resto, {}))
            escribir_tramos(p, [t for t in tramos if t[0]])
            # El quejoso se sustituye, pero el APODERADO, el acto reclamado y la
            # autoridad siguen siendo los de la plantilla. Un hueco se ve; un
            # nombre de otro asunto, no, y aquí se firma.
            avisos_ensamblado.append(
                "Revisa el resolutivo: la plantilla arrastra el apoderado, la "
                "autoridad y el acto del asunto anterior.")
    p = buscar(doc, r"^TEMA\s*:")
    if p is not None and r.tema:
        escribir(p, f"TEMA: {r.tema}")

    # ÚLTIMA RED. Si alguna marca sobrevivió a todo lo anterior —una forma que
    # el patrón no previó— se retira antes de guardar. Un «[[p.33 §1]]» con
    # corchetes dentro de una sentencia firmada es el peor final posible, y más
    # vale perder la referencia que publicarla así.
    huerfanas = 0
    for p in doc.paragraphs:
        t = texto_de(p)
        if "[[" in t:
            huerfanas += 1
            escribir(p, re.sub(r"\s*\[\[[^\]]*\]\]", "", t))
    if huerfanas:
        avisos_ensamblado.append(
            f"{huerfanas} marcas de cita no se pudieron convertir en nota al pie "
            f"y se retiraron. Esas afirmaciones quedaron sin su referencia.")

    doc.save(ruta_salida)
    return ruta_salida


# Lo que el ensamblador quiere decir sobre el documento que acaba de escribir.
# Vive fuera porque `ensamblar()` devuelve una ruta y cambiar su firma obligaría
# a tocar el endpoint y el frontend por un aviso.



def huecos_pendientes(ruta: str) -> list[str]:
    """Los párrafos que siguen sin rellenar. Nada sale con un `*****` dentro."""
    doc = docx.Document(ruta)
    fuera = []
    for p in doc.paragraphs:
        t = texto_de(p).strip()
        if HUECO in t or re.search(r"\*{4,}", t):
            fuera.append(t[:120])
        if re.match(r"^(TEMA|OPORTUNIDAD)\s*:\s*$", t):
            fuera.append(t)
    return fuera
