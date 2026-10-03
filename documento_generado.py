"""EL DOCUMENTO SIN PLANTILLA — se escribe entero, no se rellena.

David, 30-ago-2026: «El documento tiene que ser creado completamente por el
modelo, no rellenar huecos porque la herramienta no es exclusivamente para
secretarios del Tercer Tribunal Colegiado de Circuito en Materias
Administrativa y Civil del Vigésimo Segundo Circuito, sino de todo el país».

Y tiene razón por una razón que se vio medida: la plantilla NO es un formato,
es una sentencia real. Lleva dentro su tribunal, su magistrado, su quejoso, su
toca y sus fórmulas de sesión. Rellenar sus huecos funciona para quien la
escribió y produce basura para cualquier otro: un secretario de Yucatán
firmaría «Resolución del Tercer Tribunal Colegiado… del Vigésimo Segundo
Circuito» sin darse cuenta.

Aquí no hay huecos que rellenar porque no hay plantilla. Se compone:
  · lo que el asunto dice          → las fases (antecedentes, resúmenes, estudio)
  · lo que la ley obliga a decir   → lo escribe el modelo con los datos del caso
  · lo que el cómputo calcula      → la tabla, dibujada
  · lo que el tribunal es          → viene del encargo, NO del código

LO ÚNICO QUE QUEDA EN BLANCO es la fecha de la sesión, porque la sesión no ha
ocurrido. Eso no es un hueco de plantilla: es un dato que todavía no existe.

FORMATO, medido sobre los engroses reales y no inventado:
    papel oficio 21.59 × 34.03 cm · márgenes 5/2/3/3
    cuerpo Arial 14, justificado, sangría de primera línea 1.25 cm, 1.5 líneas
    cita  Arial 12, sin sangría, un espacio
"""

from __future__ import annotations

import json

# EL CATÁLOGO SE IMPORTA UNA VEZ, ARRIBA. Estaba importado dentro de las
# funciones que lo usaban, y al añadir un uso NUEVO más arriba que el
# `import` local, `_ta` quedaba sin ligar: el nombre es local a la función
# desde que aparece un import suyo en cualquier punto de ella, así que
# usarlo antes revienta con UnboundLocalError. No es un riesgo de ciclo:
# tipos_asunto no importa nada de este módulo.
import tipos_asunto as _ta

import os
import copy
import re
import unicodedata
from dataclasses import dataclass, field

import docx
from docx.enum.table import (WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT,
                             WD_ROW_HEIGHT_RULE)
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

MODELO_ESTRUCTURA = os.getenv("MODELO_ESTRUCTURA",
                              os.getenv("MODELO_ESTUDIO", "gpt-5.6-luna"))
ESFUERZO_ESTRUCTURA = os.getenv("ESFUERZO_ESTRUCTURA", "medium")
MAX_TOKENS_ESTRUCTURA = int(os.getenv("MAX_TOKENS_ESTRUCTURA", "16000"))

FUENTE = "Arial"
TAMANO = Pt(14)
# UN PUNTO MENOS QUE EL CUERPO, no dos. El adelanto ajustado escribe las citas
# a 13 sobre un cuerpo de 14: se distinguen sin empequeñecerse, que es lo que
# pasaba con 12 —la tesis quedaba en letra de nota al pie dentro del cuerpo—.
TAMANO_CITA = Pt(13)
TAMANO_TABLA = Pt(11)
SANGRIA = Cm(1.25)
# La sangría del bloque transcrito —artículo o tesis—, medida en el corpus.
SANGRIA_CITA = Cm(1.25)
# La sangría izquierda del rubro, medida en el adelanto que David ajustó.
SANGRIA_CARATULA = Cm(6.24)

# LA TESIS QUE AUTORIZA NO TRANSCRIBIR. Texto fijo, con su registro, tomado del
# adelanto que David ajustó a mano. Se escribe, no se pide: una jurisprudencia
# citada de memoria por un modelo es la alucinación que este proyecto persigue,
# y ésta es siempre la misma en los cuatro tipos.
APOYO_DISPENSA = (
    "Sustenta esa consideración, por analogía, la jurisprudencia 2a./J. "
    "58/2010 de la Segunda Sala de la Suprema Corte de Justicia de la Nación, "
    "de rubro: «CONCEPTOS DE VIOLACIÓN O AGRAVIOS. PARA CUMPLIR CON LOS "
    "PRINCIPIOS DE CONGRUENCIA Y EXHAUSTIVIDAD EN LAS SENTENCIAS DE AMPARO ES "
    "INNECESARIA SU TRANSCRIPCIÓN.»")
INTERLINEADO = 1.5
INTERLINEADO_CITA = 1.0

# Negro y gris, como pidió David. Nada de color: es una sentencia.
NEGRO = "000000"
GRIS_CABECERA = "3B3B3B"      # fondo de la fila de encabezado
GRIS_ALTERNO = "F2F2F2"       # bandeado de filas
GRIS_LINEA = "9A9A9A"         # bordes
BLANCO = "FFFFFF"


# ═══════════════════════════════════════════════════════════════════════════
# Utillaje de formato
# ═══════════════════════════════════════════════════════════════════════════

# EL AIRE ENTRE PÁRRAFOS. El adelanto ajustado separa cada párrafo del
# siguiente con un renglón entero —141 párrafos para 72 con texto—; el
# generado iba con 6 puntos, que a interlineado 1.5 y cuerpo 14 no se ve. Se
# hace con `space_after`, no metiendo párrafos vacíos: un vacío entre cada dos
# párrafos duplica el recuento, descuadra las remisiones «en el párrafo
# anterior» y le deja al secretario ciento y pico de líneas que borrar si
# quiere juntar dos ideas.
#
# LA EXCEPCIÓN ES LA CARÁTULA, que sí lleva el vacío de verdad: ahí son cuatro
# renglones y el hueco es parte de la ficha, no separación de prosa.
# CERO, Y EL AIRE SE PONE CON UN PÁRRAFO VACÍO DE VERDAD.
#
# Propuse `space_after` de 24 puntos —mismo aspecto, más sólido de editar— y
# David miró el resultado y pidió su forma: «quiero ajustar el formato
# justamente a la forma de ADELANTO_410_v2 (ajustado)». Su documento declara
# espaciado CERO y separa con 69 párrafos vacíos.
#
# Tiene razón él, y por un motivo que yo no había pesado: estos documentos se
# terminan en Word, a mano, y un renglón vacío se ve, se selecciona y se borra;
# un `space_after` de 24 puntos hay que ir a buscarlo al cuadro de párrafo. La
# solidez de un formato no es sólo cómo se genera: es cómo se deja tocar.
ESPACIO_ENTRE_PARRAFOS = Pt(0)


def _fmt(p, sangria=True, tamano=TAMANO, interlineado=INTERLINEADO,
         alineacion=WD_ALIGN_PARAGRAPH.JUSTIFY):
    pf = p.paragraph_format
    # `None` NO ES «POR OMISIÓN»: es «no la declares». Word entonces hereda la
    # del estilo, que es lo que hace el resolutivo del adelanto ajustado.
    if alineacion is not None:
        pf.alignment = alineacion
    pf.line_spacing = interlineado
    pf.space_after = ESPACIO_ENTRE_PARRAFOS
    pf.first_line_indent = SANGRIA if sangria else Cm(0)
    for r in p.runs:
        r.font.name = FUENTE
        # EL TAMAÑO SÓLO SE ESTAMPA CUANDO SE APARTA DEL ESTILO. En el
        # documento de David los párrafos no declaran tamaño: lo heredan del
        # estilo Normal, que es Arial 14. Estampar 14 en cada run parece
        # inofensivo y no lo es: el día que él cambie el cuerpo del documento
        # —a 13 para que quepa, a 12 para una versión de trabajo— el estilo
        # cambia y el texto no se mueve, porque cada párrafo lleva su tamaño
        # escrito encima. Se estampa sólo lo que de verdad es distinto: las
        # citas.
        if tamano is not None and tamano != TAMANO:
            r.font.size = tamano
    return p


def parrafo(doc, texto, sangria=True, negrita=False, tamano=TAMANO,
            interlineado=INTERLINEADO, alineacion=WD_ALIGN_PARAGRAPH.JUSTIFY):
    p = doc.add_paragraph()
    r = p.add_run(texto)
    r.bold = negrita
    return _fmt(p, sangria, tamano, interlineado, alineacion)


def tramos(doc, piezas, sangria=True, tamano=TAMANO,
           interlineado=INTERLINEADO, alineacion=WD_ALIGN_PARAGRAPH.JUSTIFY):
    """[(texto, {'bold':True}), …] en un solo párrafo."""
    p = doc.add_paragraph()
    for texto, est in piezas:
        if not texto:
            continue
        r = p.add_run(texto)
        r.bold = bool(est.get("bold"))
        r.italic = bool(est.get("italic"))
    return _fmt(p, sangria, tamano, interlineado, alineacion)


def rotulo(doc, texto, dos_puntos: bool = False):
    """«R E S U L T A N D O:» — centrado, en negrita y espaciado.

    LOS DOS PUNTOS (3-oct-2026, D7): los 8 engroses de amparo directo del banco
    de oráculo, también los de la ponencia de David, escriben «R E S U L T A N D
    O:» y «C O N S I D E R A N D O:». Con `dos_puntos`, así; sin él, como
    siempre (el camino viejo no cambia)."""
    p = doc.add_paragraph()
    r = p.add_run(" ".join(texto.upper()) + (":" if dos_puntos else ""))
    r.bold = True
    return _fmt(p, sangria=False, alineacion=WD_ALIGN_PARAGRAPH.CENTER)


def _sombrear(celda, color):
    tc = celda._tc.get_or_add_tcPr()
    sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear")
    sh.set(qn("w:color"), "auto")
    sh.set(qn("w:fill"), color)
    tc.append(sh)


def _bordes(tabla, color=GRIS_LINEA, grosor="6"):
    tbl = tabla._tbl.tblPr
    bordes = OxmlElement("w:tblBorders")
    for lado in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{lado}")
        e.set(qn("w:val"), "single")
        e.set(qn("w:sz"), grosor)
        e.set(qn("w:color"), color)
        bordes.append(e)
    tbl.append(bordes)


def _celda(celda, texto, negrita=False, color=NEGRO, fondo=None,
           alineacion=WD_ALIGN_PARAGRAPH.LEFT):
    celda.text = ""
    p = celda.paragraphs[0]
    r = p.add_run(str(texto))
    r.bold = negrita
    r.font.name = FUENTE
    r.font.size = TAMANO_TABLA
    r.font.color.rgb = RGBColor.from_string(color)
    p.paragraph_format.alignment = alineacion
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.0
    if fondo:
        _sombrear(celda, fondo)


# ═══════════════════════════════════════════════════════════════════════════
# LAS NOTAS AL PIE
# ═══════════════════════════════════════════════════════════════════════════
# Un documento nuevo NO trae la parte `word/footnotes.xml`, y el utillaje del
# ensamblador clona una nota existente de la plantilla para heredar su estilo.
# Aquí no hay plantilla, así que la parte se escribe entera: el XML, su
# relación y su tipo de contenido. Sin esto, la localización de cada tesis
# —«Gaceta S.J.F., Undécima Época, Libro 52, tomo III, página 2489»— se
# quedaba en el cuerpo o se perdía, que es lo que David detectó.

_NS_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_REL_NOTAS = ("http://schemas.openxmlformats.org/officeDocument/2006/"
              "relationships/footnotes")
_TIPO_NOTAS = ("application/vnd.openxmlformats-officedocument."
               "wordprocessingml.footnotes+xml")


def _run_llamada(parrafo, ident: int):
    """El numerito volado que llama a la nota."""
    r = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    est = OxmlElement("w:rStyle")
    est.set(qn("w:val"), "FootnoteReference")
    va = OxmlElement("w:vertAlign")
    va.set(qn("w:val"), "superscript")
    rpr.append(est)
    rpr.append(va)
    r.append(rpr)
    ref = OxmlElement("w:footnoteReference")
    ref.set(qn("w:id"), str(ident))
    r.append(ref)
    parrafo._p.append(r)


# Separa, dentro de UNA nota, los párrafos que van en renglón propio.
SEP_NOTA = "\u2029"


def _xml_notas(notas: list) -> bytes:
    """`word/footnotes.xml` con las dos notas de sistema y las nuestras."""
    def _esc(x):
        return (str(x).replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;"))
    piezas = [f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
              f'<w:footnotes xmlns:w="{_NS_W}">']
    for ident, tipo in ((-1, "separator"), (0, "continuationSeparator")):
        marca = "separator" if tipo == "separator" else "continuationSeparator"
        piezas.append(
            f'<w:footnote w:type="{tipo}" w:id="{ident}"><w:p><w:pPr>'
            f'<w:spacing w:after="0" w:line="240" w:lineRule="auto"/></w:pPr>'
            f'<w:r><w:{marca}/></w:r></w:p></w:footnote>')
    for i, texto in enumerate(notas, start=1):
        # UNA NOTA, VARIOS PÁRRAFOS: los artículos de un mismo párrafo del
        # estudio van en una sola nota (una llamada por párrafo), cada uno en
        # su renglón, separados por `SEP_NOTA` —no por «\n», que ya traen los
        # textos de las tesis de la Undécima Época y se leían como espacio—.
        # La llamada volada, sólo en el primero.
        ps = []
        for j, trozo in enumerate(str(texto).split(SEP_NOTA)):
            llamada = (f'<w:r><w:rPr><w:rStyle w:val="FootnoteReference"/>'
                       f'<w:vertAlign w:val="superscript"/></w:rPr>'
                       f'<w:footnoteRef/></w:r>') if j == 0 else ""
            ps.append(
                f'<w:p><w:pPr>'
                f'<w:spacing w:after="0" w:line="240" w:lineRule="auto"/>'
                f'<w:jc w:val="both"/></w:pPr>'
                f'{llamada}'
                f'<w:r><w:rPr><w:rFonts w:ascii="{FUENTE}" w:hAnsi="{FUENTE}"/>'
                f'<w:sz w:val="18"/></w:rPr>'
                f'<w:t xml:space="preserve"> {_esc(trozo)}</w:t></w:r>'
                f'</w:p>')
        piezas.append(f'<w:footnote w:id="{i}">' + "".join(ps) + '</w:footnote>')
    piezas.append("</w:footnotes>")
    return "".join(piezas).encode("utf8")


def _inyectar_notas(ruta: str, notas: list) -> None:
    """Mete la parte de notas en el .docx ya guardado.

    python-docx no sabe crear notas al pie, así que se añaden sobre el paquete:
    el XML, la relación desde document.xml y el Override del tipo. Se reescribe
    el zip entero porque no se puede modificar una entrada en su sitio.
    """
    import shutil
    import zipfile as _zp
    if not notas:
        return
    tmp = ruta + ".tmp"
    with _zp.ZipFile(ruta) as z:
        nombres = z.namelist()
        datos = {n: z.read(n) for n in nombres}

    datos["word/footnotes.xml"] = _xml_notas(notas)

    rels = datos["word/_rels/document.xml.rels"].decode("utf8")
    if "footnotes.xml" not in rels:
        ids = re.findall(r'Id="rId(\d+)"', rels)
        nuevo = max((int(x) for x in ids), default=0) + 1
        rels = rels.replace(
            "</Relationships>",
            f'<Relationship Id="rId{nuevo}" Type="{_REL_NOTAS}" '
            f'Target="footnotes.xml"/></Relationships>')
        datos["word/_rels/document.xml.rels"] = rels.encode("utf8")

    ct = datos["[Content_Types].xml"].decode("utf8")
    if "footnotes+xml" not in ct:
        ct = ct.replace("</Types>",
                        f'<Override PartName="/word/footnotes.xml" '
                        f'ContentType="{_TIPO_NOTAS}"/></Types>')
        datos["[Content_Types].xml"] = ct.encode("utf8")

    with _zp.ZipFile(tmp, "w", _zp.ZIP_DEFLATED) as z:
        for n in list(datos):
            z.writestr(n, datos[n])
    shutil.move(tmp, ruta)


# ═══════════════════════════════════════════════════════════════════════════
# LA CITA DE UNA TESIS
# ═══════════════════════════════════════════════════════════════════════════
# Como la dictó David y como quedó medida en su corpus:
#
#     …de rubro y texto siguientes:          ← anuncio, fin de párrafo
#     «RUBRO EN NEGRITA.»                    ← párrafo aparte
#     Texto íntegro de la tesis…             ← cursiva, sangrado, 12pt, a uno
#     ÓRGANO EMISOR.¹                        ← y ahí la nota con la localización
#
# El modelo escribe el rubro EMBEBIDO en la prosa —«…siguientes: «RUBRO.» La
# responsable…»— y así la cita queda partida y sin transcripción. Esto la
# rehace desde el ACERVO, que es de donde tiene que salir el texto: palabra por
# palabra, no de la memoria del modelo.

_RX_RUBRO = re.compile(r"[“«\"]([A-ZÁÉÍÓÚÑ][^”»\"]{18,}?)[”»\"]")


# La coleta con que el modelo cierra el anuncio, para no escribirla dos veces.
_RX_COLA_ANUNCIO = re.compile(
    r"\s*,?\s*(?:cuy[oa]s?\s+|de\s+|del\s+)?"
    r"rubro(?:\s+y\s+(?:texto|registro|contenido))?"
    r"(?:\s+(?:es|son|siguientes?|los\s+siguientes?))*\s*[:.]?\s*$", re.I)

# Un estudio invoca entre tres y seis criterios; más transcripciones que eso
# convierten la sentencia en un compendio.
MAX_CITAS_DOCUMENTO = 8


def _normaliza_rubro(x: str) -> str:
    import unicodedata
    x = unicodedata.normalize("NFKD", (x or "").upper())
    x = "".join(c for c in x if not unicodedata.combining(c))
    return re.sub(r"[^A-Z0-9]+", " ", x).strip()


_RX_REGISTRO_EN_PROSA = re.compile(r"registro(?:\s+digital)?\s*:?\s*(\d{6,7})", re.I)


def tesis_del_rubro(texto: str, tesis: list):
    """La tesis del acervo que este párrafo cita, si alguna: por su rubro o,
    si el rubro viene recortado o parafraseado, por su registro."""
    m = _RX_RUBRO.search(texto or "")
    if not m:
        # SIN RUBRO ENTRE COMILLAS: el anuncio por su registro, que es la forma
        # que enseña el propio prompt (ver `anuncio_por_registro`).
        return anuncio_por_registro(texto, tesis)
    citado = _normaliza_rubro(m.group(1))
    if len(citado) < 25:
        return anuncio_por_registro(texto, tesis)
    for t in (tesis or []):
        real = _normaliza_rubro(t.get("rubro", ""))
        if real and (real.startswith(citado[:70]) or citado.startswith(real[:70])):
            return t, m
    # POR REGISTRO. El estudio escribe «de rubro «…», registro 2007413» y el
    # rubro puede venir con una palabra cambiada: el registro no.
    for mr in _RX_REGISTRO_EN_PROSA.finditer(texto or ""):
        for t in (tesis or []):
            if str(t.get("registro") or "") == mr.group(1):
                return t, m
    # Lo entrecomillado no era un rubro del material (una constancia citada
    # con mayúscula inicial): el anuncio aún puede venir por su registro.
    h, tramo = anuncio_por_registro(texto, tesis)
    if h is not None:
        return h, tramo
    return None, m


# ═══════════════════════════════════════════════════════════════════════════
# EL ANUNCIO DE LA CITA SE COMPONE, NO SE COPIA
# ═══════════════════════════════════════════════════════════════════════════
# Una auditoría del proyecto 380/2025 encontró que TRES de las cuatro citas
# llamaban «jurisprudencia» a lo que son tesis aisladas, y una de ellas
# atribuía al Pleno como si fuera de la Primera Sala. Lo grave no es el error:
# es que el documento SE DESMENTÍA A SÍ MISMO tres párrafos después, porque la
# nota al pie —que sale del acervo— decía «[TA]; 9a. Época; Pleno». Cuerpo y
# nota, en la misma página, diciendo cosas distintas.
#
# Y la causa no era del modelo. Era MÍA, en dos sitios:
#
#   1. El prompt le daba este ejemplo literal de cómo se cita:
#         «Sirve de apoyo la jurisprudencia de la Primera Sala de la Suprema
#          Corte de Justicia de la Nación, de registro 2022074…»
#      El modelo lo copió y sólo cambió el número. Hizo lo que le pedí.
#   2. El bloque de material le enseñaba si el criterio VINCULA —«JURISPRUDENCIA
#      OBLIGATORIA» o «tesis orientadora»— pero nunca le enseñaba si es
#      jurisprudencia o tesis aislada. Dos cosas distintas que yo había fundido
#      en una etiqueta.
#
# Los dos se arreglan, pero ninguno de los dos es la defensa. La defensa es
# ésta: el anuncio se construye aquí, con los campos del acervo que ya llegan
# hasta la nota al pie. Del modelo se conserva SÓLO el verbo de enlace —«Sirve
# de apoyo», «Resulta aplicable»—, que es lo que ata la cita al razonamiento y
# es lo único que él sabe y yo no. Compuesto así el fallo es imposible: la
# frase y la nota nacen del mismo dato.

_RX_VERBO = re.compile(
    r"^(.{0,90}?)\s*(?:,\s*)?(?:resulta[n]?\s+)?(?:la|el|las|los)?\s*"
    r"(?:jurisprudencias?|tesis|criterios?|precedentes?)\b", re.I | re.S)

# Los verbos de enlace que sabemos leer. Si el modelo escribe otra cosa, se usa
# el suyo tal cual mientras no nombre instancia ni tipo; y si no hay nada
# aprovechable, «Sirve de apoyo», que es la fórmula del oficio.
_POR_DEFECTO = "Sirve de apoyo"
_RX_FORMULA_ENLACE = re.compile(
    r"^(?:sirve[n]?\s+de\s+(?:apoyo|sustento)|es\s+aplicable|son\s+aplicables|"
    r"resulta[n]?\s+aplicable[s]?|apoya[n]?\s+lo\s+anterior|ilustra[n]?|robustece[n]?|"
    r"corrobora[n]?|sustenta[n]?|cobra[n]?\s+aplicaci[óo]n|tiene[n]?\s+aplicaci[óo]n|"
    r"es\s+orientador[a]?|orienta|tambi[ée]n\s+sirve\s+de\s+apoyo|al\s+respecto|"
    r"en\s+ese\s+sentido|lo\s+anterior\s+encuentra\s+apoyo|encuentra\s+apoyo|"
    r"as[íi]\s+lo\s+(?:ha\s+)?sostenido|por\s+analog[íi]a)\b", re.I)


# Lo que puede quedar colgando al recortar el sintagma: el modelo escribe
# «Sirve de apoyo, como criterio orientador, la jurisprudencia…» y el recorte se
# lleva desde «criterio», dejando «Sirve de apoyo, como». Compuesto luego da
# «Sirve de apoyo, como, la jurisprudencia…», con la palabra huérfana entre
# comas. Se poda la cola.
_RX_COLA_HUERFANA = re.compile(
    r"[\s,;:]*\b(como|en\s+calidad\s+de|a\s+t[íi]tulo\s+de|con\s+car[áa]cter"
    r"|por|de|del|la|el|los|las|y|e)\s*$", re.I)


# ═══ EL ANUNCIO QUE CONTESTA NO SE CONVIERTE EN APOYO (AR 631/2025) ═══════
# El modelo escribía bien la distinción —«Respecto de la jurisprudencia de
# registro 2015679, de rubro «…», no resulta aplicable, porque exige una
# comparación…»— y aquí se rehacía el anuncio con la fórmula por omisión: salía
# «Sirve de apoyo la jurisprudencia… 2015679…» y, debajo del rubro, «La
# jurisprudencia en cita no resulta aplicable…». Es, palabra por palabra, la
# cita contradictoria de la Solución del 631 (verificación del 28-sep-2026,
# líneas 112-114), y la misma vuelta convertía «La recurrente invoca la
# jurisprudencia…» en «Sirve de apoyo…»: la tesis de la parte reciclada como
# apoyo del proyecto. El defecto era del compositor, no del modelo.
#
# Se conserva el arranque del modelo cuando CONTESTA el criterio: lo niega
# («no resulta aplicable», «tampoco», «resulta inaplicable»), se lo atribuye a
# quien lo invocó («la recurrente invoca», «la responsable citó») o lo toma
# como tema («respecto de», «en cuanto a»). Nunca si nombra un órgano —ése lo
# pone el acervo—, y la negación de «no obstante» o «no sólo» no cuenta.
_RX_ANUNCIO_CONTESTA = re.compile(
    r"(?:\b(?:no|tampoco|ni|sin\s+que)\b(?!\s+(?:obstante|s[óo]lo|solamente|[úu]nicamente))"
    r".{0,60}?\b(?:aplicables?|aplica[n]?|aplicaci[óo]n|apoyo|sustento|obsta[n]?|"
    r"obste[n]?|obst[áa]culo|rige[n]?|gobierna[n]?|resuelve[n]?|sostiene[n]?)\b"
    r"|\binaplicables?\b"
    r"|\b(?:la|el|los|las)\s+(?:parte\s+)?(?:recurrentes?|quejos[oa]s?|inconformes?|"
    r"terceros?(?:\s+interesad[oa]s?)?|autoridad(?:\s+responsable)?|responsable|"
    r"adherentes?|a\s+quo|juzgado(?:\s+de\s+distrito)?|juez(?:\s+de\s+distrito)?|"
    r"sala\s+responsable)\b.{0,60}?\b(?:invoc|cit|apoy|sustent|fund|transcrib|aleg|adu)\w*"
    r"|^(?:respecto|acerca|tocante|en\s+cuanto|por\s+lo\s+que\s+(?:hace|ve|toca)|"
    r"en\s+relaci[óo]n|con\s+relaci[óo]n)\b)", re.I)
_RX_VERBO_CONTESTA = re.compile(
    r"^(.{0,90}?)\s*(?:,\s*)?(?:la|el|las|los)?\s*"
    r"(?:jurisprudencias?|tesis|criterios?|precedentes?)\b", re.I | re.S)
_RX_ORGANO_EN_ANUNCIO = re.compile(
    r"\b(?:primera|segunda)\s+sala\b|\bpleno\b|suprema\s+corte|colegiad", re.I)


def es_anuncio_que_contesta(lead: str) -> bool:
    """¿El arranque del modelo contesta el criterio en vez de apoyarse en él?"""
    v = " ".join((lead or "").split())
    return bool(v) and not _RX_ORGANO_EN_ANUNCIO.search(v) \
        and bool(_RX_ANUNCIO_CONTESTA.search(v))


def _lead_que_contesta(v: str) -> str:
    """El arranque que contesta, en singular (cada cita se anuncia sola) y sin
    el artículo que queda colgando; la preposición de «respecto de» se queda."""
    v = v.strip().rstrip(",;:")
    v = re.sub(r"[\s,;:]*\b(?:la|el|los|las|y|e)\s*$", "", v, flags=re.I).strip()
    for a, b in ((r"\bresultan\b", "resulta"), (r"\baplicables\b", "aplicable"),
                 (r"\bson\b", "es"), (r"\binaplicables\b", "inaplicable"),
                 (r"\bsirven\b", "sirve"), (r"\brigen\b", "rige"),
                 (r"\bgobiernan\b", "gobierna"), (r"\bobstan\b", "obsta")):
        v = re.sub(a, b, v, flags=re.I)
    return v


def _verbo_de_enlace(anuncio: str) -> str:
    """Lo único del anuncio que escribe el modelo y merece conservarse."""
    a = " ".join((anuncio or "").split())
    if not a:
        return _POR_DEFECTO
    # En plural también —«No resultan aplicables las jurisprudencias…»—, que
    # `_RX_VERBO` no lee.
    mc = _RX_VERBO_CONTESTA.match(a)
    if mc and mc.group(1).strip() and not _RX_FORMULA_ENLACE.match(mc.group(1).strip()) \
            and es_anuncio_que_contesta(mc.group(1)):
        return _lead_que_contesta(mc.group(1))
    m = _RX_VERBO.match(a)
    if m and m.group(1).strip():
        v = m.group(1).strip().rstrip(",;:")
        # Se poda dos veces: «Sirve de apoyo, como» → «Sirve de apoyo»; y un
        # «Resulta aplicable, en calidad de» → «Resulta aplicable».
        for _ in range(2):
            v = _RX_COLA_HUERFANA.sub("", v).strip()
        # SÓLO LAS FÓRMULAS DEL OFICIO. «La misma conclusión se obtiene de la
        # jurisprudencia…» dejaba «La misma conclusión se obtiene» pegado a
        # «la jurisprudencia», sin la preposición (61/2025: «se obtiene d la
        # jurisprudencia»). Un arranque que no sea fórmula de enlace se cambia
        # por la de siempre.
        if not _RX_FORMULA_ENLACE.match(v):
            return _POR_DEFECTO
        # EN SINGULAR: cada cita se anuncia sola —«Sirven de apoyo los criterios
        # de registros 2026918 y 168958» da dos bloques, y el primero no puede
        # decir «Sirven de apoyo la jurisprudencia…» (AR 631/2025)—.
        return _lead_que_contesta(v) or _POR_DEFECTO
    # Sin sustantivo reconocible: se conserva sólo si es corto y no nombra
    # órgano ni tipo, que es lo que no puede venir de él.
    if len(a) <= 60 and not re.search(
            r"sala|pleno|colegiado|jurisprudencia|tesis aislada", a, re.I):
        return a.rstrip(" ,;:")
    return _POR_DEFECTO


def _fuerza_vincula(t: dict) -> bool:
    """¿La tesis vincula a ESTE tribunal? De `fuerza_juridica` (rediseño,
    punto 3): la jurisprudencia de otro colegiado se anuncia «como criterio
    orientador», no como la que obliga."""
    import fuerza_juridica as _fj
    return _fj.vincula(t) is True


def anuncio_de(t: dict, anuncio_del_modelo: str = "") -> str:
    """«Sirve de apoyo la tesis aislada del Pleno…, de registro 191358».

    Cada pieza sale del acervo. `tipo` decide el sustantivo, `instancia` el
    órgano y `obligatoria` el calificativo —esa parte YA funcionaba: las tres
    aisladas iban «como criterio orientador» y la única jurisprudencia real «de
    carácter obligatorio»—; lo que fallaba era el sustantivo y el nombre del
    órgano, que venían del modelo.
    """
    tipo = str(t.get("tipo") or "").strip().upper()
    inst = " ".join(str(t.get("instancia") or "").strip().split())
    reg = str(t.get("registro") or "").strip()
    verbo = _verbo_de_enlace(anuncio_del_modelo)

    if "AISLAD" in tipo:
        sustantivo = "la tesis aislada"
    elif "JURISPRUDENCIA" in tipo:
        sustantivo = "la jurisprudencia"
    else:
        # SIN EL DATO NO SE AFIRMA NINGUNO DE LOS DOS. «El criterio» es cierto
        # de la jurisprudencia y de la tesis aislada, así que no miente; decir
        # «jurisprudencia» sin saberlo, sí.
        sustantivo = "el criterio"

    # EL ORDEN ES EL DEL OFICIO, no el que salga. Un tribunal escribe «Sirve de
    # apoyo, como criterio orientador, la tesis aislada del Pleno de la Suprema
    # Corte de Justicia de la Nación, de registro digital 191358, de rubro y
    # texto siguientes:». El calificativo va entre comas detrás del verbo, no
    # colgando del órgano, donde suena a que el órgano es el orientador.
    # EL CALIFICATIVO SÓLO CUANDO APORTA. En una jurisprudencia la
    # obligatoriedad va de suyo y escribirla suena a énfasis de quien no está
    # seguro; en una tesis aislada, en cambio, decir que sólo orienta es
    # información necesaria y es lo que evita que se lea como vinculante.
    # EL QUE CONTESTA NO LLEVA «COMO CRITERIO ORIENTADOR»: «No resulta
    # aplicable, como criterio orientador, la tesis aislada…» es un
    # contrasentido. Lo que se distingue no orienta nada.
    if es_anuncio_que_contesta(verbo):
        frase = f"{verbo} {sustantivo}"
    elif _fuerza_vincula(t):
        # Un verbo con inciso propio —«Es aplicable, además»— pide cerrar la
        # coma antes del sustantivo, o queda «además la jurisprudencia».
        frase = (f"{verbo}, {sustantivo}" if "," in verbo
                 else f"{verbo} {sustantivo}")
    else:
        frase = f"{verbo}, como criterio orientador, {sustantivo}"
    if inst:
        # El artículo, según el órgano. «del Tribunales Colegiados» no es
        # español: los órganos en plural piden «de los».
        if re.match(r"^(primera|segunda|tercera|cuarta)\s+sala", inst, re.I):
            de = "de la"
        elif re.search(r"^(tribunales|plenos|salas)\b", inst.strip(), re.I):
            de = "de los" if not inst.strip().lower().startswith("salas") else "de las"
        else:
            de = "del"
        # El Pleno y las Salas son de la Corte y así se nombran en un engrose;
        # los colegiados y los plenos regionales traen su nombre completo en el
        # propio campo y no se les añade nada.
        completo = inst
        if re.match(r"^(pleno|primera\s+sala|segunda\s+sala)$", inst.strip(), re.I):
            completo = f"{inst} de la Suprema Corte de Justicia de la Nación"
        frase += f" {de} {completo}"
    if reg:
        frase += f", de registro digital {reg}"
    # ── SE ANUNCIA LO QUE SE VA A ENTREGAR, NI MÁS ─────────────────────────
    # Decía siempre «de rubro y texto siguientes:» y debajo aparecía sólo el
    # rubro. No es que el texto se pierda —`escribir_cita` lo baja a la nota al
    # pie cuando pasa de MAX_PALABRAS_TESIS_CUERPO, que es una decisión tomada
    # a propósito para que el estudio no quede sepultado bajo sus citas—, pero
    # el anuncio seguía prometiendo que venía a continuación. En una sentencia
    # eso es un defecto de forma que el magistrado devuelve.
    #
    # Medido en el 650-2025: los CUATRO bloques anunciaban «rubro y texto» y en
    # los cuatro sólo bajaba el rubro.
    #
    # La condición es exactamente la misma que usa `escribir_cita` para decidir
    # dónde va el texto. Si cambia una, tiene que cambiar la otra.
    _cuerpo = _sin_coletilla_de_organo(t.get("texto") or "")
    if _cuerpo and len(_cuerpo.split()) <= MAX_PALABRAS_TESIS_CUERPO:
        return frase + ", de rubro y texto siguientes:"
    return frase + ", de rubro siguiente:"


# ═══ EL ANUNCIO SIN RUBRO SE RECONOCE POR SU REGISTRO (AR 631/2025) ════════
# El prompt enseña, como la forma de citar, «Sirve de apoyo el criterio de
# registro 2022074:» —sin tipo, sin órgano y sin rubro: «el documento los pone
# solo»—, y `tesis_del_rubro` sólo reconocía una cita si traía el rubro entre
# comillas: el registro era un respaldo DENTRO de ese camino, nunca una puerta
# propia. En la rama de coherencia (f4f88b7, 28-sep-2026) el prompt dejó de
# empujar al modelo a escribir el anuncio entero («LA INSTANCIA VA SIEMPRE»
# pasó a «LA INSTANCIA, FUERA DEL ANUNCIO»), el modelo obedeció el ejemplo al
# pie de la letra y las tres citas del estudio salieron como prosa: «Sirve de
# apoyo el criterio de registro 2026918:» y, debajo, nada —ni rubro, ni texto,
# ni nota—.
#
# Se reconoce el anuncio por su FORMA, no por la cifra: el registro es de una
# tesis del material y la oración que lo contiene ARRANCA con un verbo de
# enlace (`_RX_FORMULA_ENLACE`) o con un arranque que contesta
# (`es_anuncio_que_contesta`), con el sustantivo —criterio, tesis,
# jurisprudencia, precedente— delante del registro. Una mención en mitad de un
# razonamiento —«El criterio con registro 188480, invocado…, tampoco conduce…»,
# «…y los criterios con registros 187528, 2015679…»— no arranca así y sigue
# siendo prosa.
_RX_REGISTRO_ANUNCIO = re.compile(
    r"\bregistros?(?:\s+digital(?:es)?)?\s*(?:n[úu]mero\s+)?:?\s*(\d{6,7})\b", re.I)
# Los demás de la misma lista: «registros 2026918 y 168958», «2026918, 168958».
_RX_LISTA_REGISTROS = re.compile(
    r"(?:\s*(?:,|y|e)\s*(?:(?:el|la)\s+(?:de\s+)?)?(?:registro\s+(?:digital\s+)?)?\d{6,7}\b)*", re.I)
# Una oración acaba en punto seguido de mayúscula. «1a./J. 74/2005», «S.J.F. y»
# o «Pág. 590» no la acaban. Los dos puntos tampoco: «de rubro y texto
# siguientes: «RUBRO»» es la misma oración.
_RX_FRONTERA_ORACION = re.compile(r"\.\s+(?=[«\"“¿(]?[A-ZÁÉÍÓÚÑ])")
MAX_ARRANQUE_ANUNCIO = 220
_RX_NIEGA = re.compile(r"\b(?:no|tampoco|ni|sin\s+que)\b|inaplicable", re.I)
_RX_FIN_DE_ANUNCIO = re.compile(
    r"\s*(?:[:.]|$|,?\s*(?:porque|pues|ya\s+que|toda\s+vez\s+que|dado\s+que|"
    r"puesto\s+que|en\s+virtud\s+de\s+que)\b)", re.I)


class _Tramo:
    """Lo que `_escribir_estudio` usa de un `re.Match`: dónde empieza y dónde
    acaba lo que el bloque de la cita sustituye."""

    def __init__(self, a: int, b: int):
        self._a, self._b = a, b

    def start(self) -> int:
        return self._a

    def end(self) -> int:
        return self._b


def _inicio_de_oracion(texto: str, pos: int) -> int:
    ini = 0
    for m in _RX_FRONTERA_ORACION.finditer(texto or "", 0, pos):
        ini = m.end()
    return ini


def _arranque_de_cita(trozo: str) -> str:
    """El arranque que ata una cita —«Sirve de apoyo», «No resulta aplicable»,
    «La recurrente invoca»—, si el trozo que precede al registro o al rubro lo
    es; si no, cadena vacía."""
    v = " ".join((trozo or "").split())
    mc = _RX_VERBO_CONTESTA.match(v)
    if not mc:
        return ""
    a = mc.group(1).strip(" ,;:")
    if a and (_RX_FORMULA_ENLACE.match(a) or es_anuncio_que_contesta(a)):
        return a
    return ""


def anuncio_por_registro(texto: str, tesis: list):
    """(tesis, tramo) del anuncio que cita por su registro y sin rubro; o
    (None, None). El tramo va del registro al último de su lista."""
    t = texto or ""
    por_reg = {str(x.get("registro") or "").strip(): x for x in (tesis or [])
               if str(x.get("registro") or "").strip()}
    if not por_reg:
        return None, None
    for mr in _RX_REGISTRO_ANUNCIO.finditer(t):
        h = por_reg.get(mr.group(1))
        if h is None:
            continue
        ini = _inicio_de_oracion(t, mr.start())
        lead = t[ini:mr.start()]
        a = _arranque_de_cita(lead) if len(lead) <= MAX_ARRANQUE_ANUNCIO else ""
        if not a:
            continue
        fin = _RX_LISTA_REGISTROS.match(t, mr.end()).end()
        # EL ARRANQUE QUE SÓLO ATRIBUYE —«La recurrente invoca el criterio de
        # registro N para sostener que…», «Respecto del criterio de registro N,
        # invocado por…»— es un anuncio sólo si la oración acaba ahí o sigue
        # con su razón («, porque…»). Si continúa contando qué pretendía la
        # parte, es prosa del estudio y así se queda.
        if not (_RX_FORMULA_ENLACE.match(a) or _RX_NIEGA.search(a)) \
                and not _RX_FIN_DE_ANUNCIO.match(t[fin:]):
            continue
        return h, _Tramo(mr.start(), fin)
    return None, None


def registros_de_la_lista(texto: str, tramo, tesis: list, fuera: str = "") -> list:
    """Los OTROS registros del material en la lista de un mismo anuncio
    («Sirven de apoyo los criterios de registros 2026918 y 168958:»)."""
    if tramo is None or isinstance(tramo, re.Match):
        return []
    tengo = {str(x.get("registro") or "").strip() for x in (tesis or [])}
    vistos, otros = {fuera}, []
    for r in re.findall(r"\d{6,7}", (texto or "")[tramo.start():tramo.end()]):
        if r in tengo and r not in vistos:
            vistos.add(r)
            otros.append(r)
    return otros


def escribir_cita(doc, t: dict, anuncio: str, notas: list) -> None:
    """El bloque entero de la cita, con su nota al pie."""
    # EL ANUNCIO NO SE COPIA: se compone del acervo. Ver `anuncio_de`.
    parrafo(doc, anuncio_de(t, anuncio))

    # El rubro, solo y en negrita. Sin párrafo vacío delante: la cita va
    # pegada a su anuncio y el aire lo pone el espaciado.
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    r = p.add_run(f"«{(t.get('rubro') or '').strip().rstrip('.')}.»")
    r.bold = True
    _fmt(p, sangria=False, tamano=TAMANO_CITA, interlineado=INTERLINEADO_CITA)
    p.paragraph_format.left_indent = Cm(1.25)
    # EL RUBRO SÓLO SE ATA A ALGO SI HAY ALGO DEBAJO.
    # `keep_with_next` existe para que el rubro no quede huérfano del texto de
    # su tesis. Cuando ese texto baja a la nota al pie —que es lo normal, son
    # tesis de trescientas palabras— debajo del rubro no queda nada suyo, y
    # atarlo al párrafo siguiente hace que Word empuje los dos a la página
    # próxima en cuanto la nota ocupa el pie. Es el salto en blanco de media
    # hoja de la revisión fiscal 2/2026: página 28, el anuncio arriba, quince
    # centímetros vacíos y el rubro solo al principio de la 29.
    # Se decide abajo, cuando ya se sabe si el texto se queda o baja.
    _rubro = p

    # EL TEXTO DE LA TESIS BAJA A LA NOTA AL PIE. Iba íntegro en el cuerpo, y
    # eso es lo que hace que un estudio con seis criterios invocados quede
    # sepultado bajo sus propias citas: medido en los engroses de referencia,
    # las tres transcripciones de tesis del ARA 17/2025 se llevan la mitad del
    # considerando, y en la queja 233/2025 la transcripción de una ejecutoria
    # de la Corte ocupa 2,260 de 3,798 palabras.
    #
    # El rubro se queda arriba —identifica el criterio y se lee de un vistazo—
    # y el texto va abajo, donde quien firma lo comprueba si quiere. Es un
    # cambio sobre el corpus, no una imitación suya, y está pedido: «nuestro
    # redactor debe ser mejor que el secretario que redactó esos proyectos».
    #
    # SE CONSERVA EN EL CUERPO CUANDO ES CORTO. Una tesis de cuatro renglones
    # leída al pie es una molestia sin ganancia; el problema son las de
    # trescientas palabras.
    # LO QUE QUEDA EN EL CUERPO BASTA PARA QUE LA EXPLICACIÓN SE ENTIENDA (AR
    # 631/2025, 28-sep-2026): el anuncio con su registro digital —lo pone
    # `anuncio_de`— y el rubro; el texto, sólo en la nota. La regla la dice el
    # estudio con sus palabras antes o después de la cita, y `_sin_eco` ya no
    # la borra cuando el texto está al pie (ver `UMBRAL_ECO_AL_PIE`).
    cuerpo = _sin_coletilla_de_organo(t.get("texto") or "")
    _al_pie = len(cuerpo.split()) > MAX_PALABRAS_TESIS_CUERPO
    # Atado sólo si su texto va debajo; si va al pie, el rubro fluye y el aire
    # lo pone su espaciado: un renglón, no media página.
    _rubro.paragraph_format.keep_with_next = bool(cuerpo and not _al_pie)
    if cuerpo and not _al_pie:
        q = doc.add_paragraph()
        rq = q.add_run(cuerpo)
        rq.italic = True
        _fmt(q, sangria=False, tamano=TAMANO_CITA,
             interlineado=INTERLINEADO_CITA)
        q.paragraph_format.left_indent = Cm(1.25)
        # EL BLOQUE LARGO NO SE ATA A LO SIGUIENTE. Con `keep_with_next` en
        # toda la cadena —anuncio, rubro, texto y órgano— Word empuja el
        # conjunto entero a la página siguiente y deja media hoja en blanco:
        # ése era el «espacio enorme antes de citar una tesis». El rubro sí
        # sigue atado a su texto, que es lo que no puede partirse.
        q.paragraph_format.keep_with_next = False

    # El órgano, y ahí cuelga la nota con la localización.
    inst = (t.get("instancia") or "").strip()
    loc = (t.get("localizacion") or "").strip()
    reg = str(t.get("registro") or "").strip()
    # ═══════════════════════════════════════════════════════════════════════
    # EL ÓRGANO VA EN LA NOTA, NO EN UN RENGLÓN SUYO
    # ═══════════════════════════════════════════════════════════════════════
    # Se escribía como párrafo aparte, en negrita y con la llamada de la nota
    # colgando de él: debajo de cada rubro aparecía un renglón suelto que decía
    # «SEGUNDA SALA.» y nada más.
    #
    # David lo borró a mano en las cuatro tesis de su revisión 650/2025, y el
    # corpus le da la razón sin margen: de los 1,358 documentos de la carpeta
    # del tribunal, UNO escribe el órgano en línea aparte. No es el estilo de
    # la casa; era una invención nuestra.
    #
    # EL DATO NO SE PIERDE: se va al principio de la nota, junto a la
    # localización y el registro, que es donde ya vivía el resto de la ficha.
    # Y la llamada pasa a colgar del rubro, que es lo que se está citando.
    if inst or loc or reg or _al_pie:
        pie = ", ".join(x for x in (inst.strip().rstrip("."), loc) if x)
        if reg and reg not in pie:
            pie = (pie + ", " if pie else "") + f"registro digital {reg}"
        if _al_pie:
            pie = (pie + ". " if pie else "") + f"Texto: {cuerpo}"
        # LA LLAMADA CUELGA DEL RUBRO. `p` es el párrafo del rubro, que sigue
        # existiendo; el que desaparece es el del órgano.
        if pie in notas:
            _run_llamada(p, notas.index(pie) + 1)
        else:
            notas.append(pie)
            _run_llamada(p, len(notas))


# ═══════════════════════════════════════════════════════════════════════════
# LA TABLA DEL CÓMPUTO
# ═══════════════════════════════════════════════════════════════════════════

# EL ÓRGANO EMISOR VIENE PEGADO AL FINAL DEL TEXTO DE LA TESIS. Así lo publica
# el Semanario y así está guardado en el acervo: «…viola el citado principio.
# PRIMER TRIBUNAL COLEGIADO DEL OCTAVO CIRCUITO.». Al bajar el texto a la nota
# al pie, esa coletilla quedaba como un párrafo suelto en mayúsculas detrás de
# la transcripción —visto en la revisión fiscal 2/2026—, y encima contradice a
# la instancia que la propia ficha ya declara dos líneas más arriba. El dato no
# se pierde: la ficha lo dice mejor y en su sitio.
_RX_ORGANO_AL_FINAL = re.compile(
    r"\s*\n?\s*((?:PRIMER|SEGUNDO|TERCER|CUARTO|QUINTO|SEXTO|S[ÉE]PTIMO|OCTAVO|"
    r"NOVENO|D[ÉE]CIMO|[IVXL]+)[^.\n]{0,120}TRIBUNAL[^.\n]{0,120}|"
    r"(?:PRIMERA|SEGUNDA)\s+SALA[^.\n]{0,60}|PLENO[^.\n]{0,80}|"
    r"TRIBUNALES\s+COLEGIADOS[^.\n]{0,80})\.\s*$")


def _sin_coletilla_de_organo(texto: str) -> str:
    """El texto de la tesis, sin el órgano emisor pegado al final."""
    t = (texto or "").rstrip()
    for _ in range(2):          # algunas traen órgano y «Esta tesis se publicó…»
        nuevo = _RX_ORGANO_AL_FINAL.sub("", t).rstrip()
        if nuevo == t:
            break
        t = nuevo
    return t


def _sin_partir(tabla) -> None:
    """`cantSplit`: la fila entera va a la página donde quepa."""
    from docx.oxml.ns import qn
    for fila in tabla.rows:
        trPr = fila._tr.get_or_add_trPr()
        el = trPr.makeelement(qn("w:cantSplit"), {})
        trPr.append(el)


# ═══════════════════════════════════════════════════════════════════════════
# EL CALENDARIO DEL CÓMPUTO
# ═══════════════════════════════════════════════════════════════════════════
# David: «un calendario (moderno y profesional) que sustituya la tabla en tonos
# grises que tenemos (parece aburrida)… el cómputo ocurre en dos meses (no
# siempre en uno) por lo que serían dos calendarios».
#
# LA TABLA DE HITOS NO SE VA: se queda debajo, porque es la que se cita —«el
# plazo corrió del doce al diecinueve»— y la que se copia al engrose. Lo que
# hace el calendario es lo que ninguna lista de fechas hace: enseñar de un
# vistazo POR QUÉ el plazo terminó ese día y no otro. Los fines de semana y los
# inhábiles se ven como huecos, y el lector cuenta los cuadros llenos.
#
# LA PALETA ES SOBRIA A PROPÓSITO. Esto se imprime, se fotocopia y a veces se
# escanea en blanco y negro, así que cada marca se distingue TAMBIÉN por el
# tono de gris que deja al perder el color, y ninguna depende de un rojo o un
# verde que un daltónico no separaría. El día del plazo va en azul pizarra
# sobre blanco; los tres hitos, en pleno con el número en blanco.
AZUL_PLAZO = "DCE4EC"       # los días hábiles que corrieron
AZUL_BORDE = "2E4A62"       # el pleno de los hitos
VERDE_HITO = "1E6B4F"       # presentación: el que cierra
AMBAR_HITO = "8A5A1B"       # notificación y surtimiento: los que abren
GRIS_FUERA = "F7F7F7"       # días de otro mes o inhábiles
GRIS_TENUE = "9A9A9A"       # su número

_DIAS_SEMANA = ("L", "M", "M", "J", "V", "S", "D")
_MESES = ("enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
          "agosto", "septiembre", "octubre", "noviembre", "diciembre")


def _meses_del_computo(computo) -> list:
    """[(año, mes)] que hay que dibujar, del primero al último hito."""
    import datetime as _d
    hitos = [computo.notificacion, computo.surtio, computo.inicio]
    if not getattr(computo, "en_cualquier_tiempo", False):
        hitos.append(computo.vencimiento)
    if computo.presentacion is not None:
        hitos.append(computo.presentacion)
    hitos = [h for h in hitos if h]
    if not hitos:
        return []
    ini, fin = min(hitos), max(hitos)
    meses, cur = [], _d.date(ini.year, ini.month, 1)
    while (cur.year, cur.month) <= (fin.year, fin.month):
        meses.append((cur.year, cur.month))
        cur = _d.date(cur.year + (cur.month == 12), cur.month % 12 + 1, 1)
        # TRES MESES SON YA UNA AGENDA, no un cómputo. Pasa con los plazos de
        # treinta días o cuando media un periodo vacacional; se dibujan los
        # tres, pero más allá el calendario deja de aclarar y estorba, y la
        # tabla de hitos sigue estando debajo.
        if len(meses) >= 4:
            break
    return meses


# ═══════════════════════════════════════════════════════════════════════════
# EL CALENDARIO QUE SE EXPLICA SOLO (25-sep-2026)
# ═══════════════════════════════════════════════════════════════════════════
# David, con el 93/2026 delante: «hay que darle lógica al calendario. Indicar al
# lector qué implican los días marcados en rojo, qué son los días en gris,
# marcar en negro cuando se presenta la demanda (con la indicación en letras
# pequeñas en el propio recuadro) (…) y rediseñar la tabla de abajo a un mapa
# visual mejorado, más moderno (no tabla simple) que ilustre el plazo».
#
# Tres piezas, y cada una contesta una pregunta del lector:
#   · EL CALENDARIO — ¿qué pasó cada día? Cada recuadro marcado dice qué es,
#     en letra pequeña dentro de él: «notificación», «surte efectos», «día 7»,
#     «inhábil», «presentación».
#   · LA LEYENDA — ¿qué significa cada color? Con su muestra y su porqué.
#   · EL MAPA — ¿cómo se llega de la notificación a la presentación? Cuatro
#     tarjetas enlazadas, la barra de los días hábiles y el veredicto.
#
# SIGUE SIENDO LEGIBLE EN BLANCO Y NEGRO: la presentación va en negro pleno,
# los hitos en ámbar oscuro, los días del plazo en gris azulado claro y los que
# no corren en blanco; y cada marca lleva además su palabra, que es lo que no
# se pierde en una fotocopia.
NEGRO_HITO = "1A1A1A"       # presentación: el día que se busca
ROSA_FUERA = "EFD3CE"       # hábiles que corrieron después del vencimiento
ROJO_VEREDICTO = "8B2A1E"   # fuera de plazo
FONDO_TARJETA = "F5F7F9"
TAMANO_ETIQUETA = Pt(6.5)
TAMANO_LEYENDA = Pt(9)
ANCHO_UTIL = 14.5           # cm: oficio con 5 cm de margen izquierdo y 2 derecho

_MESES_CORTOS = ("ene", "feb", "mar", "abr", "may", "jun", "jul", "ago",
                 "sep", "oct", "nov", "dic")
_DIAS_LARGOS = ("lunes", "martes", "miércoles", "jueves", "viernes",
                "sábado", "domingo")


def _fecha_corta(f) -> str:
    return f"{f.day} {_MESES_CORTOS[f.month - 1]} {f.year}" if f else ""


def _fund_corto(f: str) -> str:
    """«artículo 65 de la Ley Federal de Procedimiento Contencioso
    Administrativo» → «art. 65 LFPCA». Para la tarjeta, no para el
    considerando, que lo lleva entero."""
    t = str(f or "")
    t = re.sub(r",?\s+de\s+la\s+Ley\s+Federal\s+de\s+Procedimiento\s+Contencioso\s+"
               r"Administrativo", " LFPCA", t)
    t = re.sub(r",?\s+de\s+la\s+Ley\s+de\s+Amparo", " LA", t)
    t = re.sub(r"\bart[íi]culos\b", "arts.", t)
    t = re.sub(r"\bart[íi]culo\b", "art.", t)
    t = re.sub(r"\bfracci[óo]n\b", "fr.", t)
    return t.strip()


def _sin_bordes(tabla) -> None:
    tbl = tabla._tbl.tblPr
    bordes = OxmlElement("w:tblBorders")
    for lado in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{lado}")
        e.set(qn("w:val"), "nil")
        bordes.append(e)
    tbl.append(bordes)


def _disposicion_fija(tabla, margen_cm: float = None) -> None:
    """Anchos que Word respeta, y márgenes de celda a la medida."""
    tbl = tabla._tbl.tblPr
    lay = OxmlElement("w:tblLayout")
    lay.set(qn("w:type"), "fixed")
    tbl.append(lay)
    if margen_cm is not None:
        mar = OxmlElement("w:tblCellMar")
        for lado in ("left", "right"):
            e = OxmlElement(f"w:{lado}")
            e.set(qn("w:w"), str(int(margen_cm * 567)))
            e.set(qn("w:type"), "dxa")
            mar.append(e)
        tbl.append(mar)


def _borde_celda(celda, **lados) -> None:
    """_borde_celda(c, top=("single", 18, "8A5A1B"), left=("nil",))"""
    tcPr = celda._tc.get_or_add_tcPr()
    tcb = OxmlElement("w:tcBorders")
    for lado, spec in lados.items():
        e = OxmlElement(f"w:{lado}")
        e.set(qn("w:val"), spec[0])
        if len(spec) > 1:
            e.set(qn("w:sz"), str(spec[1]))
            e.set(qn("w:space"), "0")
            e.set(qn("w:color"), spec[2])
        tcb.append(e)
    tcPr.append(tcb)


def _anchos(tabla, cms: list) -> None:
    """El ancho en la rejilla Y en cada celda: Word lee la celda, pero otros
    visores —y Word al reabrir con otra impresora— leen la rejilla, y con
    sólo una de las dos la leyenda salía con las muestras hechas hilo."""
    for col, w in zip(tabla.columns, cms):
        col.width = Cm(w)
    for fila in tabla.rows:
        for c, w in zip(fila.cells, cms):
            c.width = Cm(w)


def _run(p, texto, tamano=TAMANO_TABLA, negrita=False, color=NEGRO):
    r = p.add_run(str(texto))
    r.bold = negrita
    r.font.name = FUENTE
    r.font.size = tamano
    r.font.color.rgb = RGBColor.from_string(color)
    return r


def _p_compacto(p, alineacion=WD_ALIGN_PARAGRAPH.CENTER):
    p.paragraph_format.alignment = alineacion
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.0
    return p


def _celda_dia(celda, dia, fondo=None, color=NEGRO, negrita=False,
               etiquetas=(), color_etiqueta=GRIS_TENUE) -> None:
    """El número del día y, debajo, en letra pequeña, lo que ese día es."""
    celda.text = ""
    celda.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    p = _p_compacto(celda.paragraphs[0])
    _run(p, dia, negrita=negrita, color=color)
    for et in etiquetas:
        _run(_p_compacto(celda.add_paragraph()), et, tamano=TAMANO_ETIQUETA,
             color=color_etiqueta)
    if fondo:
        _sombrear(celda, fondo)


def _dias_del_computo(computo) -> dict:
    """{fecha: (clase, [etiquetas])} para cada día que el calendario explica.

    Clases: notificacion · surte · plazo · presentacion · fuera · inhabil.
    La presentación va la última: si cae en un día del plazo —lo normal— es
    ese dato el que se busca, y lleva además su número de día."""
    import datetime as _d
    d = {}
    numero = {f: i for i, f in enumerate(computo.dias or [], 1)}
    total = len(computo.dias or [])
    for f, i in numero.items():
        ets = [f"día {i}"]
        if i == total and not getattr(computo, "en_cualquier_tiempo", False):
            ets = [f"día {i} · vence"]
        d[f] = ("plazo", ets)
    # Lo que NO corrió entre la notificación y el final: se dice por qué.
    ini = computo.notificacion
    fin = max([x for x in (computo.vencimiento, computo.presentacion) if x] or [ini])
    cur = ini
    while ini and cur <= fin:
        if cur not in d and cur.weekday() < 5 and not computo.cal_amparo.es_habil(cur):
            d[cur] = ("inhabil", ["inhábil"])
        cur += _d.timedelta(days=1)
    for f in getattr(computo, "resp_dias", None) or []:
        if ini and ini <= f <= fin and f not in numero and f.weekday() < 5:
            d[f] = ("inhabil", ["sin labores"])
    # Hábiles después del vencimiento: la distancia a la presentación, a la vista.
    if (computo.presentacion is not None and computo.oportuna is False
            and not getattr(computo, "en_cualquier_tiempo", False)):
        k, cur = 0, computo.vencimiento + _d.timedelta(days=1)
        while cur <= computo.presentacion:
            if computo.cal_amparo.es_habil(cur):
                k += 1
                d[cur] = ("fuera", [f"fuera +{k}"])
            cur += _d.timedelta(days=1)
    if computo.notificacion:
        d[computo.notificacion] = ("notificacion", ["notificación"])
    if computo.surtio:
        if computo.surtio == computo.notificacion:
            d[computo.surtio] = ("notificacion", ["notificación", "y surte efectos"])
        else:
            d[computo.surtio] = ("surte", ["surte efectos"])
    if computo.presentacion is not None:
        ets = ["presentación"]
        if computo.presentacion in numero:
            ets.append(f"día {numero[computo.presentacion]}")
        elif computo.presentacion in d and d[computo.presentacion][0] == "fuera":
            ets.append(d[computo.presentacion][1][0])
        d[computo.presentacion] = ("presentacion", ets)
    return d


def _calendario_mes(doc, anio, mes, computo, dias) -> None:
    """Un mes, con cada día marcado diciendo qué es."""
    import calendar as _c, datetime as _d
    _c.setfirstweekday(_c.MONDAY)
    semanas = _c.monthcalendar(anio, mes)

    t = doc.add_table(rows=2, cols=7)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    _disposicion_fija(t, margen_cm=0.05)
    _bordes(t, color="D8D8D8", grosor="4")

    cab = t.rows[0].cells
    cab[0].merge(cab[6])
    _celda(t.rows[0].cells[0], f"{_MESES[mes - 1].upper()} {anio}",
           negrita=True, color=BLANCO, fondo=AZUL_BORDE,
           alineacion=WD_ALIGN_PARAGRAPH.CENTER)
    for i, d in enumerate(_DIAS_SEMANA):
        _celda(t.rows[1].cells[i], d, negrita=True, color=GRIS_TENUE,
               alineacion=WD_ALIGN_PARAGRAPH.CENTER)

    for semana in semanas:
        fila = t.add_row()
        fila.height = Cm(0.95)
        fila.height_rule = WD_ROW_HEIGHT_RULE.AT_LEAST
        for i, dia in enumerate(semana):
            c = fila.cells[i]
            if dia == 0:
                _celda(c, "", fondo=GRIS_FUERA)
                continue
            f = _d.date(anio, mes, dia)
            clase, ets = dias.get(f, ("", []))
            if clase == "presentacion":
                _celda_dia(c, dia, fondo=NEGRO_HITO, color=BLANCO, negrita=True,
                           etiquetas=ets, color_etiqueta=BLANCO)
            elif clase in ("notificacion", "surte"):
                _celda_dia(c, dia, fondo=AMBAR_HITO, color=BLANCO, negrita=True,
                           etiquetas=ets, color_etiqueta=BLANCO)
            elif clase == "plazo":
                _celda_dia(c, dia, fondo=AZUL_PLAZO, negrita=True,
                           etiquetas=ets, color_etiqueta=AZUL_BORDE)
            elif clase == "fuera":
                _celda_dia(c, dia, fondo=ROSA_FUERA, negrita=True,
                           etiquetas=ets, color_etiqueta=ROJO_VEREDICTO)
            elif clase == "inhabil":
                _celda_dia(c, dia, color=GRIS_TENUE, etiquetas=ets)
            else:
                _celda_dia(c, dia, color=GRIS_TENUE)
    _anchos(t, [ANCHO_UTIL / 7] * 7)
    _sin_partir(t)


def _precepto_de_la_tabla(regla, tipo_asunto, papel: str = "",
                          fundamento_surtimiento: str = "") -> str:
    """El precepto del surtimiento que dice la TABLA, el mismo del párrafo.

    QUIÉN RECURRE ENTRA AQUÍ TAMBIÉN (3-oct-2026, regresión sin bandera): el
    párrafo de la oportunidad ya no cita la fracción II del 31 cuando recurre
    una autoridad, pero la leyenda y la tarjeta del cómputo la pedían sin el
    papel y seguían imprimiendo «art. 31, fr. II LA»: el mismo .docx decía dos
    cosas. Y el precepto que el secretario declaró, si lo declaró."""
    import fase0_oportunidad as _f0t
    f = _f0t.fundamento_de_surtimiento(regla, tipo_asunto, papel)
    if f:
        return f
    d = " ".join(str(fundamento_surtimiento or "").split()).strip(" .")
    if d and HUECO not in d:
        return re.sub(r"^(?:conforme\s+a(?:l|\s+los?)?|en\s+t[ée]rminos\s+del?|el|los)\s+", "",
                      d, flags=re.I)
    return ""


def _leyenda_computo(doc, computo, tipo_asunto, dias, papel: str = "",
                     fundamento_surtimiento: str = "", forma_consta: bool = True) -> None:
    """Cada color con su muestra y su porqué: lo que un color sin nombre no dice.

    Con `forma_consta=False` (F1, quinta ronda) no se dice CÓMO se notificó,
    como en el párrafo: «La notificación surte efectos al día hábil siguiente…»."""
    import fase0_oportunidad as _f0l
    _v = _ta.vocabulario_de(tipo_asunto)
    _escrito = _v["escrito"]
    reg = computo.regla
    f_surte = _precepto_de_la_tabla(reg, tipo_asunto, papel, fundamento_surtimiento)
    # EN LOS RECURSOS NO HAY «LEY DEL ACTO»: lo notificado es del juicio de
    # amparo; si el precepto no se puede decir (recurre una autoridad con la
    # regla de los particulares), el hueco, como en el párrafo.
    _sin_f = (f" ({HUECO})" if _ta.normalizar(tipo_asunto) in ("amparo_revision", "queja")
              else " (ley del acto)")
    f_ini = _f0l.fundamento_de_inicio(tipo_asunto)
    surte_txt = _f0l._ORDINAL_SURTE.get(getattr(reg, "dias_habiles", 1), "al día hábil siguiente")
    filas = []
    if computo.notificacion:
        if getattr(reg, "clave", "") == "otra":
            porque = "fecha de surtimiento declarada por el promovente"
        else:
            porque = (f"la notificación{(' ' + reg.descripcion) if forma_consta else ''} "
                      f"surte efectos {surte_txt}"
                      + (f" ({_fund_corto(f_surte)})" if f_surte else _sin_f))
            # EL 227-I DICE CUÁNDO CORREN LOS TÉRMINOS, NO CUÁNDO SURTE (revisión
            # de normas, 3-oct-2026): la leyenda dice lo que dice la ley, como el
            # párrafo (`fase0_oportunidad.parrafo_oportunidad`).
            if getattr(reg, "clave", "") == "cnpcf_personal" and f_surte:
                porque = (f"la notificación{(' ' + reg.descripcion) if forma_consta else ''} "
                          f"surte efectos {surte_txt}, pues los términos corren desde el día "
                          f"siguiente al de la notificación personal ({_fund_corto(f_surte)})")
        filas.append((AMBAR_HITO, "Notificación y día en que surtió efectos",
                       porque[0].upper() + porque[1:] + "."))
    if not getattr(computo, "en_cualquier_tiempo", False):
        filas.append((AZUL_PLAZO, f"Días hábiles del plazo, numerados del 1 al {computo.plazo}",
                      "Corre a partir del día siguiente al en que surtió efectos la notificación"
                      + (f" ({_fund_corto(f_ini)})" if f_ini else "") + "."))
    if computo.presentacion is not None:
        if computo.anticipada:
            q = "antes de que empezara a correr el plazo, lo que no le resta oportunidad"
        elif computo.oportuna is False:
            q = "después del vencimiento"
        elif computo.presentacion == computo.vencimiento:
            q = "el último día del plazo"
        else:
            n = (computo.dias or []).index(computo.presentacion) + 1 \
                if computo.presentacion in (computo.dias or []) else None
            q = f"el día {n} de {computo.plazo}" if n else "dentro del plazo"
        filas.append((NEGRO_HITO, f"Presentación {_ta_del(_escrito)}", q[0].upper() + q[1:] + "."))
    if any(c == "fuera" for c, _ in dias.values()):
        filas.append((ROSA_FUERA, "Días hábiles transcurridos después del vencimiento",
                      "Cada uno lleva su cuenta: «fuera +1», «fuera +2»…"))
    _no = "Sábados y domingos"
    _tr = _f0l.tramos_inhabiles(computo.inhabiles_en_medio, computo.cal_amparo)
    if _tr and isinstance(_tr, _f0l.TramosInhabiles):
        # CADA TRAMO CON SU FUNDAMENTO (3-oct-2026): la leyenda decía «(art. 19
        # LA)» también del dos de mayo de una circular o de las vacaciones.
        _partes_no = []
        for _k, _ts in _f0l.grupos_inhabiles(_tr):
            _corto = (_fund_corto(computo.cal_amparo.fundamento) if _k == "art19"
                      else _f0l.FUENTES_INHABIL.get(_k, {}).get("corto", ""))
            _partes_no.append("; ".join(
                (_fecha_corta(a) if a == b else f"{_fecha_corta(a)} a {_fecha_corta(b)}")
                for a, b in _ts) + (f" ({_corto})" if _corto else ""))
        _no += " (" + _fund_corto(computo.cal_amparo.fundamento) + "); " + "; ".join(_partes_no)
    elif _tr:
        _no += "; " + "; ".join(
            (_fecha_corta(a) if a == b else f"{_fecha_corta(a)} a {_fecha_corta(b)}")
            for a, b in _tr) + f" ({_fund_corto(computo.cal_amparo.fundamento)})"
    if getattr(computo, "resp_tramos_en_medio", None):
        _no += "; días sin labores de la responsable: " + "; ".join(
            (_fecha_corta(a) if a == b else f"{_fecha_corta(a)} a {_fecha_corta(b)}")
            for a, b in computo.resp_tramos_en_medio)
    filas.append((BLANCO, "Sin color: días que no corren", _no + "."))

    # UN CUADRO DE COLOR POR LÍNEA, no una columna sombreada: sombreadas, las
    # muestras se juntaban en una sola barra y no se sabía cuál era de quién.
    # El cuadro es un carácter (■ / □), así que también sale en la fotocopia.
    t = doc.add_table(rows=0, cols=2)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    _sin_bordes(t)
    _disposicion_fija(t, margen_cm=0.05)
    for color, titulo, detalle in filas:
        fila = t.add_row()
        m, x = fila.cells
        m.text = ""
        m.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.TOP
        _glifo = "□" if color == BLANCO else "■"
        _col = ("9A9A9A" if color == BLANCO else "9FB3C8" if color == AZUL_PLAZO else color)
        _run(_p_compacto(m.paragraphs[0]), _glifo, tamano=Pt(13), color=_col)
        x.text = ""
        p_ = _p_compacto(x.paragraphs[0], WD_ALIGN_PARAGRAPH.LEFT)
        p_.paragraph_format.space_before = Pt(2)
        p_.paragraph_format.space_after = Pt(3)
        _run(p_, titulo + ". ", tamano=TAMANO_LEYENDA, negrita=True, color="3B3B3B")
        _run(p_, detalle, tamano=TAMANO_LEYENDA, color="5A5A5A")
    _anchos(t, [0.6, ANCHO_UTIL - 0.6])
    _sin_partir(t)


def _ta_del(x: str) -> str:
    from fase0_oportunidad import _del as _d0
    return _d0(x)


def calendario_computo(doc, computo, tipo_asunto: str = "amparo_directo", papel: str = "",
                       fundamento_surtimiento: str = "", forma_consta: bool = True) -> None:
    """Los meses del cómputo, cada día marcado diciendo qué es, y su leyenda."""
    meses = _meses_del_computo(computo)
    if not meses:
        return
    dias = _dias_del_computo(computo)
    for anio, mes in meses:
        _calendario_mes(doc, anio, mes, computo, dias)
        parrafo(doc, "", sangria=False)
    _leyenda_computo(doc, computo, tipo_asunto, dias, papel, fundamento_surtimiento,
                     forma_consta=forma_consta)
    parrafo(doc, "", sangria=False)


def _fundamentos_juntos(a: str, b: str) -> str:
    """«art. 17 LA» + «art. 18 LA» → «arts. 17 y 18 LA»."""
    ma = re.match(r"^art\. (\S+) (LA|LFPCA)$", a or "")
    mb = re.match(r"^art\. (\S+) (LA|LFPCA)$", b or "")
    if ma and mb and ma.group(2) == mb.group(2):
        return f"arts. {ma.group(1)} y {mb.group(1)} {ma.group(2)}"
    return " · ".join(x for x in (a, b) if x)


def _tarjeta(celda, rotulo, color, grande, lineas) -> None:
    """Una tarjeta del mapa: franja de color arriba, rótulo, dato y detalle."""
    celda.text = ""
    celda.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.TOP
    _sombrear(celda, FONDO_TARJETA)
    _borde_celda(celda, top=("single", 24, color), left=("nil",), right=("nil",),
                 bottom=("single", 4, "D8DEE4"))
    p = _p_compacto(celda.paragraphs[0])
    p.paragraph_format.space_before = Pt(3)
    _run(p, rotulo, tamano=Pt(7.5), negrita=True, color=color if color != AZUL_PLAZO else AZUL_BORDE)
    p = _p_compacto(celda.add_paragraph())
    p.paragraph_format.space_before = Pt(2)
    _run(p, grande, tamano=Pt(11.5), negrita=True, color=NEGRO)
    for i, (txt, col) in enumerate(lineas):
        p = _p_compacto(celda.add_paragraph())
        if i == len(lineas) - 1:
            p.paragraph_format.space_after = Pt(3)
        _run(p, txt, tamano=Pt(7.5), color=col)


def _flecha(celda) -> None:
    celda.text = ""
    celda.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    _run(_p_compacto(celda.paragraphs[0]), "›", tamano=Pt(20), negrita=True, color="B5BEC7")


def _habiles_de_retraso(computo) -> int:
    """Días hábiles entre el vencimiento y la presentación, ésta incluida."""
    import datetime as _d
    if computo.presentacion is None or computo.oportuna is not False:
        return 0
    k, cur = 0, computo.vencimiento + _d.timedelta(days=1)
    while cur <= computo.presentacion:
        if computo.cal_amparo.es_habil(cur):
            k += 1
        cur += _d.timedelta(days=1)
    return k


def _dias_habiles_txt(k: int) -> str:
    return f"{k} día hábil" if k == 1 else f"{k} días hábiles"


def _junto_al_siguiente(tabla) -> None:
    """Que el mapa no se parta: cada párrafo de la tabla pide ir con el que
    sigue. Sin esto el veredicto se iba solo a la hoja siguiente."""
    for fila in tabla.rows:
        for c in fila.cells:
            for p in c.paragraphs:
                p.paragraph_format.keep_with_next = True


def mapa_computo(doc, computo, fecha_en_letra=None,
                 tipo_asunto: str = "amparo_directo", papel: str = "",
                 fundamento_surtimiento: str = "", forma_consta: bool = True) -> None:
    """EL MAPA DEL PLAZO: de la notificación a la presentación, de un vistazo.

    Sustituye a la tabla de dos columnas «concepto · fecha». Las fechas exactas
    siguen aquí —y en letra, en el considerando, que es lo que se copia al
    engrose—; lo que cambia es que el lector ve el recorrido: qué abrió el
    plazo, cuántos días hábiles tuvo, dónde cayó la presentación y el resultado.
    Todo con tablas, sombreados y bordes de Word: se edita como cualquier otra
    tabla del documento."""
    import fase0_oportunidad as _f0m
    _v = _ta.vocabulario_de(tipo_asunto)
    reg = computo.regla
    sin_plazo = bool(getattr(computo, "en_cualquier_tiempo", False))
    f_surte = _precepto_de_la_tabla(reg, tipo_asunto, papel, fundamento_surtimiento)
    f_ini = _f0m.fundamento_de_inicio(tipo_asunto)
    f_plazo = _ta.plazo_de(tipo_asunto, "").get("fundamento") or ""

    def _dia(f):
        return _DIAS_LARGOS[f.weekday()] if f else ""

    medio = re.sub(r"^(?:mediante|de manera|por)\s+", "", str(reg.descripcion or "")).strip()
    # LA FORMA QUE NO CONSTA TAMPOCO SE AFIRMA EN LA TARJETA (F1, quinta ronda).
    if not forma_consta:
        medio = "forma no consta"
    tarjetas = [
        ("NOTIFICACIÓN", AMBAR_HITO, _fecha_corta(computo.notificacion),
         [(_dia(computo.notificacion), GRIS_TENUE),
          ((medio[:1].upper() + medio[1:]) if medio else "", "5A5A5A")]),
    ]
    if getattr(reg, "clave", "") == "otra":
        _det = "fecha declarada"
    else:
        _det = {1: "día hábil siguiente", 2: "2.º día hábil", 3: "3.er día hábil"}.get(
            reg.dias_habiles, "mismo día" if reg.dias_habiles == 0 else f"{reg.dias_habiles}.º día hábil")
    tarjetas.append(("SURTE EFECTOS", AMBAR_HITO, _fecha_corta(computo.surtio),
                     [(_dia(computo.surtio), GRIS_TENUE),
                      (_det + (f" · {_fund_corto(f_surte)}" if f_surte else ""), "5A5A5A")]))
    if sin_plazo:
        tarjetas.append(("PLAZO", AZUL_PLAZO, "Sin plazo",
                         [("procede en cualquier tiempo", "5A5A5A")]))
    else:
        tarjetas.append(("PLAZO", AZUL_PLAZO, f"{_fecha_corta(computo.inicio)}",
                         [(f"al {_fecha_corta(computo.vencimiento)}", NEGRO),
                          (f"{computo.plazo} días hábiles · "
                           + _fundamentos_juntos(_fund_corto(f_plazo), _fund_corto(f_ini)),
                           "5A5A5A")]))
    if computo.presentacion is not None:
        if computo.anticipada:
            _q = "antes de iniciar el plazo"
        elif computo.presentacion in (computo.dias or []):
            _q = f"día {(computo.dias or []).index(computo.presentacion) + 1} de {computo.plazo}"
        elif computo.oportuna is False:
            _q = f"{_dias_habiles_txt(_habiles_de_retraso(computo))} después del vencimiento"
        else:
            _q = ""
        tarjetas.append(("PRESENTACIÓN", NEGRO_HITO, _fecha_corta(computo.presentacion),
                         [(_dia(computo.presentacion), GRIS_TENUE), (_q, "5A5A5A")]))
    elif not sin_plazo:
        tarjetas.append(("VENCIMIENTO", AZUL_BORDE, _fecha_corta(computo.vencimiento),
                         [(_dia(computo.vencimiento), GRIS_TENUE)]))

    # ── 1. LAS TARJETAS, ENLAZADAS ─────────────────────────────────────────
    n = len(tarjetas)
    flecha_w = 0.55
    tarj_w = (ANCHO_UTIL - flecha_w * (n - 1)) / n
    t = doc.add_table(rows=1, cols=2 * n - 1)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    _sin_bordes(t)
    _disposicion_fija(t, margen_cm=0.1)
    celdas = t.rows[0].cells
    for i, (rot, col, grande, lineas) in enumerate(tarjetas):
        _tarjeta(celdas[2 * i], rot, col, grande, [x for x in lineas if x[0]])
        if i < n - 1:
            _flecha(celdas[2 * i + 1])
    _anchos(t, [tarj_w if j % 2 == 0 else flecha_w for j in range(2 * n - 1)])
    _sin_partir(t)
    _junto_al_siguiente(t)

    # ── 2. LA BARRA DE LOS DÍAS HÁBILES ────────────────────────────────────
    if not sin_plazo and computo.dias:
        dias = _dias_del_computo(computo)
        fuera = sorted(f for f, (c, _) in dias.items() if c == "fuera")
        anticipada = bool(computo.anticipada)
        # La presentación tardía cierra la barra: sin ella el segmento negro
        # —el dato que se busca— no aparecía cuando más importa.
        tardia = ([computo.presentacion] if (computo.presentacion is not None
                  and computo.oportuna is False) else [])
        segmentos = (["pre"] if anticipada else []) + list(computo.dias) + fuera[:10] + tardia
        m = len(segmentos)
        pp = parrafo(doc, "", sangria=False)
        pp.paragraph_format.space_after = Pt(0)
        pp.paragraph_format.keep_with_next = True
        b = doc.add_table(rows=2, cols=m)
        b.alignment = WD_TABLE_ALIGNMENT.CENTER
        b.autofit = False
        _sin_bordes(b)
        _disposicion_fija(b, margen_cm=0)
        fila = b.rows[0]
        fila.height = Cm(0.42)
        fila.height_rule = WD_ROW_HEIGHT_RULE.EXACTLY
        for j, s in enumerate(segmentos):
            c = fila.cells[j]
            c.text = ""
            if s == "pre":
                color = NEGRO_HITO
            elif s == computo.presentacion:
                color = NEGRO_HITO
            elif s in fuera:
                color = ROSA_FUERA
            else:
                color = "9FB3C8"
            _sombrear(c, color)
            _borde_celda(c, left=("single", 8, "FFFFFF"), right=("single", 8, "FFFFFF"))
        # Debajo, los extremos: dónde empezó y dónde terminó.
        abajo = b.rows[1].cells
        mitad = max(1, m // 2)
        izq = abajo[0].merge(abajo[mitad - 1]) if mitad > 1 else abajo[0]
        der = abajo[mitad].merge(abajo[m - 1]) if m - mitad > 1 else abajo[mitad]
        izq.text = der.text = ""
        _run(_p_compacto(izq.paragraphs[0], WD_ALIGN_PARAGRAPH.LEFT),
             f"día 1 · {_fecha_corta(computo.inicio)}", tamano=Pt(7.5), color="5A5A5A")
        if computo.presentacion is not None and not anticipada:
            _fin_txt = (f"día {computo.plazo} · {_fecha_corta(computo.vencimiento)}"
                        if computo.presentacion == computo.vencimiento
                        else f"vence · {_fecha_corta(computo.vencimiento)}   "
                             f"presentación · {_fecha_corta(computo.presentacion)}")
        else:
            _fin_txt = f"vence · {_fecha_corta(computo.vencimiento)}"
        _run(_p_compacto(der.paragraphs[0], WD_ALIGN_PARAGRAPH.RIGHT),
             _fin_txt, tamano=Pt(7.5), color="5A5A5A")
        _anchos(b, [ANCHO_UTIL / m] * m)
        _sin_partir(b)
        _junto_al_siguiente(b)

    # ── 3. EL VEREDICTO ────────────────────────────────────────────────────
    if computo.oportuna is not None:
        if getattr(computo, "rectificada", False):
            fondo, txt = AZUL_BORDE, ("EL CÓMPUTO ARROJA FUERA DE PLAZO · EL TRIBUNAL LO TIENE "
                                      "POR OPORTUNO POR LA RAZÓN QUE EXPRESA EL CONSIDERANDO")
        elif computo.anticipada:
            fondo, txt = VERDE_HITO, "EN TIEMPO · PRESENTADA ANTES DEL INICIO DEL PLAZO"
        elif computo.oportuna:
            fondo = VERDE_HITO
            txt = ("EN TIEMPO · PRESENTADA EL ÚLTIMO DÍA DEL PLAZO"
                   if computo.presentacion == computo.vencimiento else "EN TIEMPO")
        else:
            _k = _habiles_de_retraso(computo)
            fondo, txt = ROJO_VEREDICTO, ("FUERA DE PLAZO" + (
                f" · PRESENTADA {_dias_habiles_txt(_k).upper()} DESPUÉS DEL VENCIMIENTO"
                if _k else ""))
        pp = parrafo(doc, "", sangria=False)
        pp.paragraph_format.space_after = Pt(0)
        pp.paragraph_format.keep_with_next = True
        v = doc.add_table(rows=1, cols=1)
        v.alignment = WD_TABLE_ALIGNMENT.CENTER
        v.autofit = False
        _sin_bordes(v)
        c = v.rows[0].cells[0]
        c.text = ""
        c.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        _sombrear(c, fondo)
        p = _p_compacto(c.paragraphs[0])
        p.paragraph_format.space_before = Pt(3)
        p.paragraph_format.space_after = Pt(3)
        _run(p, txt, tamano=Pt(9.5), negrita=True, color=BLANCO)
        _anchos(v, [ANCHO_UTIL])
        _sin_partir(v)

    # AIRE ANTES DEL SIGUIENTE CONSIDERANDO: sin él, el «CUARTO.» quedaba
    # pegado a la franja del veredicto (93/2026, en producción).
    parrafo(doc, "", sangria=False)


# El nombre viejo, por si algo externo lo llama: ya no hay tabla de dos columnas.
tabla_computo = mapa_computo


# ═══════════════════════════════════════════════════════════════════════════
# Lo que la ley obliga a decir — lo escribe el modelo, con los datos del caso
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Estructura:
    apertura: str = ""
    visto: str = ""
    resultandos: list = field(default_factory=list)   # [{"titulo","texto"}]
    competencia: str = ""
    existencia: str = ""
    procedencia: str = ""
    avisos: list = field(default_factory=list)
    # LOS AVISOS QUE PUSO LA ÚLTIMA COMPOSICIÓN (28-sep-2026). `componer` cuelga
    # sus avisos de `avisos` para que viajen de vuelta, y la estructura se
    # reutiliza del adelanto al proyecto: sin separarlos, los del adelanto
    # —compuesto sin criterio, «rama confirma_sobresee: … el recurso resultó
    # infundado»— salían en el proyecto junto a los de éste («rama
    # revoca_fondo_niega»). AR 631/2025. Cada composición retira los de la
    # anterior antes de poner los suyos.
    avisos_de_composicion: list = field(default_factory=list)


# LOS AVISOS DE LA RAMA DE LA REVISIÓN: dicen qué hizo el juzgado y qué punto se
# escribió por eso. Si el proyecto los dice, los del adelanto sobran (ver
# `redactor_adelanto._terminar`). Por su arranque, que es fijo.
_AVISOS_DE_RAMA = (
    "RESOLUTIVO DE REVISIÓN, rama",
    "EL SENTIDO DE LA SENTENCIA RECURRIDA NO SE PUDO LEER DEL PDF",
    "LO QUE HIZO EL JUZGADO SE TOMÓ DE SU PUNTO RESOLUTIVO",
    "EL SEGUNDO RESOLUTIVO NIEGA LO QUE EL JUZGADO CONCEDIÓ",
    "EL PRIMER RESOLUTIVO DICE «EN LA MATERIA DE LA REVISIÓN»",
    "SE ASUME JURISDICCIÓN Y EL SEGUNDO RESOLUTIVO",
    "SE REVOCA LA CONCESIÓN Y SE REASUME JURISDICCIÓN",
    "SE REVOCA UNA CONCESIÓN Y LOS CONCEPTOS",
    "NO CONSTA QUÉ RESOLVIÓ EL JUZGADO",
    "LA SENTENCIA RECURRIDA ES MIXTA",
    "LA SENTENCIA RECURRIDA TAMBIÉN SOBRESEYÓ",
    "LOS CONCEPTOS DE VIOLACIÓN NO ESTUDIADOS",
)


def es_aviso_de_rama(aviso) -> bool:
    """¿Este aviso habla de la rama de la revisión (qué hizo el juzgado y qué
    resolutivo salió de ahí)?"""
    return str(aviso or "").startswith(_AVISOS_DE_RAMA)


_RX_JSON = re.compile(r"\{.*\}", re.S)


# CADA ASUNTO SE IDENTIFICA CON LO QUE TIENE, y no todos tienen lo mismo. El
# prompt ordenaba identificar el acto por «fecha, SALA, TOCA y expediente de
# origen, y qué confirmó, modificó o revocó»: eso es un amparo directo contra
# una sentencia de segunda instancia. En una QUEJA se recurre un auto de un juez
# de distrito y en una REVISIÓN una sentencia de amparo indirecto: no hay sala,
# no hay toca y no hay nada que confirmar ni revocar. El modelo, obligado a
# decirlo, escribía «dentro del toca y expediente de origen que constan en
# autos, acto que no confirmó, modificó ni revocó otra resolución», que es la
# frase defensiva que un dictamen ya nos reprochó: una sentencia no explica al
# lector por qué NO hay toca; sencillamente no lo menciona.
_IDENTIFICA_ACTO = {
    "amparo_directo": ("fecha, sala, toca y expediente de origen, y qué "
                       "confirmó, modificó o revocó"),
    "amparo_revision": ("fecha, juzgado de distrito y número del juicio de "
                        "amparo indirecto en que se dictó"),
    "revision_fiscal": ("fecha, sala del Tribunal Federal de Justicia "
                        "Administrativa y número del juicio de nulidad"),
    "queja": ("fecha, juzgado de distrito y número del juicio de amparo en que "
              "se dictó, y qué proveyó"),
}
# Y EL AMPARO DIRECTO SIN ALZADA (30-sep-2026). La fórmula de arriba es la de
# una sentencia de segunda instancia, y en un juicio oral mercantil, un laudo o
# una sentencia de nulidad no hay sala de apelación, ni toca, ni nada que
# confirmar o revocar (AD 323/2025). La regla es la de este mismo comentario:
# no se explica por qué no hay toca, sencillamente no se menciona. Se usa sólo
# si la instancia consta como única (`tipos_asunto.unica_instancia`).
_IDENTIFICA_ACTO_UNICA = ("fecha, órgano que la dictó y número del expediente "
                          "de origen, y qué resolvió en ese juicio: si declaró "
                          "procedente o improcedente la acción, si condenó o "
                          "absolvió, si declaró la nulidad o reconoció la "
                          "validez")
_NOMBRE_ASUNTO = {
    "amparo_directo": "amparo directo",
    "amparo_revision": "amparo en revisión",
    "revision_fiscal": "revisión fiscal",
    "queja": "recurso de queja",
}


def prompt_estructura(datos: dict) -> str:
    q = "agravios" if datos.get("es_recurso") else "conceptos de violación"
    _tipo = str(datos.get("tipo_asunto") or "amparo_directo").strip().lower()
    _clase = _NOMBRE_ASUNTO.get(_tipo, "amparo directo")
    # LOS RESULTANDOS SON DEL TIPO. Estaban escritos a mano aquí —los cuatro
    # del amparo directo— y salían iguales en una queja y en una revisión
    # fiscal. Ahora se arman del catálogo, que es donde están medidos.
    import tipos_asunto as _ta_r
    _rs = []
    for _i, (_rot, _que) in enumerate(_ta_r.resultandos_de(_tipo)):
        # EL DATO QUE FALTA SE REMITE AL EXPEDIENTE; NO SE OMITE NI SE LAMENTA.
        #
        # Esto decía «se omite y ya», y por eso el resultando del turno salió
        # con «No consta en los datos proporcionados la fecha en que el asunto
        # fue turnado»: el modelo ni lo omitió ni lo remitió, se quejó. Y
        # «los datos proporcionados» es el material del prompt, no el
        # expediente que el lector tiene delante.
        #
        # David: «hay que tratar de producir un proyecto completo en la medida
        # de lo posible, y si no hay datos, remitirnos al expediente, salvo en
        # lo relativo a las fechas de sesión y de lista, que esas
        # necesariamente serán incorporadas por el secretario».
        # «EN LOS TÉRMINOS QUE OBRAN EN AUTOS» SE QUITA (3-oct-2026): el
        # prompt la proponía y `tipos_asunto._EVASIVAS` la denuncia; el modelo
        # la escribía y el documento se acusaba a sí mismo.
        _extra = (" Si algún dato de éstos no lo encuentras, NO digas que no "
                  "consta ni menciones «los datos proporcionados»: escribe la "
                  "frase completa remitiendo al expediente —«en la fecha que "
                  "se advierte de las constancias»—, que es lo que hace un "
                  "secretario cuando el dato está en el expediente y no a la "
                  "vista. El resultando tiene que quedar COMPLETO y legible."
                  + ("" if _i else
                     " PROHIBIDO resumir aquí su razonamiento: eso va en el "
                     "estudio."))
        # UNA LLAVE, NO DOS (3-oct-2026). Esto no es un f-string: con «{{» el
        # prompt enseñaba llaves dobles literales —`{{"titulo": …}}`— en un
        # ejemplo de JSON que el modelo tiene que imitar.
        _rs.append('     {"titulo": %s, "texto": "<%s>"}'
                   % (json.dumps(_rot, ensure_ascii=False), _que + _extra))
    _resultandos = ",\n".join(_rs)

    # LA FÓRMULA DEL PROEMIO SALE DEL CATÁLOGO, NO DE UN EJEMPLO. Este campo
    # traía ESCRITA la del amparo directo —«para resolver el juicio de amparo
    # directo…»— y un modelo con un ejemplo concreto delante lo copia y le
    # cambia los datos: de ahí salían «V I S T O, para resolver el juicio de
    # amparo directo relacionado con el recurso de queja civil» y el mismo
    # apócrifo en la revisión fiscal. Es la tercera vez en este proyecto que un
    # ejemplo del prompt se firma literal —antes fue «la jurisprudencia de la
    # Primera Sala» y «si lo traes, es por analogía»—, así que la regla ya no
    # es una sospecha: en un prompt, lo que se escribe entero se copia entero.
    _molde_visto = _ta_r.proemio_de(_tipo)["molde"]

    # LA HOJA DE DATOS TAMBIÉN ENSEÑABA EL AMPARO DIRECTO. Le decía al motor
    # «QUEJOSO: …» y «AUTORIDAD RESPONSABLE: …» en los cuatro tipos, así que
    # aunque los rótulos de los resultandos ya vinieran bien del catálogo, la
    # PROSA seguía hablando de quejoso y de autoridad responsable dentro de una
    # queja. Era el canal por el que la plantilla única volvía a entrar después
    # de haberla quitado de la estructura.
    _vc = _ta_r.vocabulario_de(_tipo)
    # «QUEJOSA» PARA LA PERSONA MORAL. La etiqueta del catálogo va en
    # masculino; con la parte separada de su representante ya se sabe si es
    # una sociedad (o el nombre es femenino) y la carátula lo dice bien.
    def _etiqueta_parte(_et: str, _cl: str) -> str:
        if _cl != "quejoso" or not _et.upper().startswith("QUEJOSO"):
            return _et
        _fem = bool(datos.get("quejoso_moral")) or \
            _ta_r.genero_de(str(datos.get("quejoso") or "")) == "a"
        return _et.replace("QUEJOSO", "QUEJOSA", 1) if _fem else _et
    # ═══ QUIÉN RECURRE, DICHO APARTE (23-sep-2026) ═══════════════════════
    # 711/2025: la hoja de datos decía «QUEJOSA Y RECURRENTE: [la sociedad]» y
    # el VISTO salió «recurso de revisión interpuesto por Interamericana…»
    # cuando lo interpuso la UIF. Con `recurrente` aparte, la hoja lo dice en
    # su renglón y el molde del VISTO recibe al que de verdad recurrió.
    _recurrente_hoja = str(datos.get("recurrente") or "").strip()
    # LA MISMA CARÁTULA QUE EL .docx (`tipos_asunto.filas_caratula`): el
    # recurrente con su carácter en su renglón y, en la revisión, sin el
    # juzgado en el rubro (0 de 478 en el corpus; AR 631/2025). El juzgado y
    # la responsable del acto van como DATOS aparte, sin rótulo de carátula.
    _filas_hoja = []
    for _et, _cl, _val in _ta_r.filas_caratula(_tipo, datos):
        # El género ya viene concordado (con `quejoso_moral` sólo si recurre la
        # propia quejosa): `_etiqueta_parte` se aplica sólo en ese caso, el de
        # antes, para no feminizar a la quejosa por la forma de la recurrente.
        if _cl == "quejoso" and not _recurrente_hoja:
            _et = _etiqueta_parte(_et, _cl)
        if _cl == "recurrente":
            _et += " (quien interpuso el recurso)"
        _filas_hoja.append(f"{_et}: {_val or ''}")
    if _ta_r.normalizar(_tipo) == "amparo_revision":
        _org_hoja = str(datos.get("organo_recurrido") or "").strip()
        if _org_hoja:
            _filas_hoja.append(f"Dato, no va en la carátula · juzgado que dictó la "
                               f"sentencia recurrida: {_org_hoja}")
        if str(datos.get("responsable") or "").strip():
            _filas_hoja.append(f"Dato, no va en la carátula · autoridad responsable "
                               f"del acto reclamado en el amparo: {datos.get('responsable')}")
    elif _ta_r.normalizar(_tipo) in ("queja", "revision_fiscal") and not any(
            _cl_h == "responsable" for _e_h, _cl_h, _o_h in _ta_r.caratula_de(_tipo)):
        # C3 Y C4 (3-oct-2026): el órgano de la queja y la Sala de la revisión
        # fiscal ya no van en el rubro (David: «hay que quitar»; 0 de 8 quejas y
        # 28 de 28 revisiones fiscales del banco sin ellos); la hoja los sigue
        # dando como dato, que el V I S T O y la competencia los nombran.
        # LA RESPONSABLE DEL FORMULARIO, SÓLO SI TIENE LA FORMA DEL ÓRGANO
        # (integración, 3-oct-2026): sin el renglón, la pantalla manda ahí la
        # ordenadora del auto de admisión (`tipos_asunto.responsable_es_el_organo`).
        _resp_h = str(datos.get("responsable") or "").strip()
        # CON LA FRACCIÓN DEL 97, COMO LAS OTRAS DOS LLAMADAS (revisión Q, 3-oct-
        # 2026): en la queja de la fracción II quien dictó el auto ES la
        # responsable del amparo directo, una Sala, y sin la fracción la hoja
        # perdía su renglón.
        _fr97_h = ""
        for _src_h in (datos.get("procesal"), datos.get("tramite")):
            if isinstance(_src_h, dict) and str(_src_h.get("fraccion_97") or "").strip():
                _fr97_h = str(_src_h.get("fraccion_97")).strip()
                break
        _org_hoja = str(datos.get("organo_recurrido") or "").strip() or (
            _resp_h if _ta_r.responsable_es_el_organo(_tipo, _resp_h, _fr97_h) else "")
        if _org_hoja:
            _que_h = ("órgano que dictó el auto recurrido" if _ta_r.normalizar(_tipo) == "queja"
                      else "Sala que dictó la sentencia recurrida")
            _filas_hoja.append(f"Dato, no va en la carátula · {_que_h}: {_org_hoja}")
    _ficha_partes = "\n".join(_filas_hoja)
    if _recurrente_hoja:
        _ficha_partes += ("\nOJO: el recurso lo interpuso el RECURRENTE, no la quejosa. En el "
                          "VISTO y en los resultandos, «interpuesto por» lleva al recurrente.")
    _rotulo_acto = {"amparo_directo": "ACTO RECLAMADO"}.get(
        _tipo, _vc["sub_recurrido"].upper())
    _de_escrito = ("DE LA DEMANDA" if _tipo == "amparo_directo"
                   else "DEL " + _vc["escrito"].upper())

    # ERA CÓDIGO MUERTO Y ERA JUSTO LA INSTRUCCIÓN QUE FALTABA: se calculaba y
    # no se interpolaba en ningún sitio del prompt. Dice cómo se identifica el
    # acto en cada tipo —«fecha, sala, toca y expediente de origen» en el
    # amparo directo— y ahora entra donde sirve, junto a la ficha de datos.
    _identifica = _IDENTIFICA_ACTO.get(_tipo, _IDENTIFICA_ACTO["amparo_directo"])
    # SIN ALZADA (30-sep-2026): ni sala, ni toca, ni «qué confirmó» — ver
    # `_IDENTIFICA_ACTO_UNICA`. Y la regla de no inventar ponía de ejemplo «un
    # número de toca», que en única instancia es pedirle que lo busque.
    _dato_ejemplo = "un número de toca"
    if _ta_r.unica_instancia(_tipo):
        _identifica = _IDENTIFICA_ACTO_UNICA
        _dato_ejemplo = "un número de expediente"
    # LA FICHA PROCESAL, COMO DATOS (SPEC_E2, 28-sep-2026): quién promovió,
    # quién recurre y con qué carácter, qué órgano dictó la recurrida y qué
    # resolvió por acto, con su fuente. Es de donde salen la carátula, la
    # competencia y la legitimación; en el AR 631/2025 el prompt sólo tenía los
    # renglones de la hoja y escribió «dictada … por la MAGISTRADA». Vacía, el
    # prompt queda como estaba.
    _ficha_bloque = str(datos.get("ficha_bloque") or "").rstrip()
    if _ficha_bloque:
        _ficha_partes += "\n" + _ficha_bloque
    return f"""Eres el secretario de un Tribunal Colegiado de Circuito y escribes las
partes ESTRUCTURALES de una sentencia de {_clase}. No escribes el estudio
de fondo —ese ya está hecho—: escribes lo que la ley obliga a decir antes de
llegar a él, con los datos de ESTE asunto y de ESTE tribunal.

EL TRIBUNAL QUE RESUELVE: {datos.get('tribunal','')}
CIUDAD: {datos.get('ciudad','')}
EXPEDIENTE: {datos.get('encabezado','')}
{_ficha_partes}
{_rotulo_acto} —léelo de aquí e IDENTIFÍCALO por {_identifica}—:
{(datos.get('acto') or '').strip() or '(no se aportó: NO lo describas, omite la mención)'}
FECHA DE PRESENTACIÓN {_de_escrito}: {datos.get('presentacion','')}
MAGISTRADO PONENTE: {datos.get('magistrado','')}
SECRETARIO: {datos.get('secretario','')}

ANTECEDENTES DEL ASUNTO, ya redactados
{datos.get('antecedentes','')[:40000]}

REGLAS:
- ESTOS SON LOS ROTULOS MEDIDOS EN LOS ENGROSES DEL CORPUS, no una
  propuesta: escríbelos tal cual. El resultando lleva esos {len(_rs)} apartados
  y el documento añade solo el de la sesión.
- NO ESCRIBAS ORDINALES. Nada de «PRIMERO.» ni «SEGUNDO.»: el documento los
  calcula y ponerlos aquí los duplica.
- NO INVENTES DATOS. Si no sabes una fecha, {_dato_ejemplo} o un nombre, NO
  lo pongas y NO lo sustituyas por uno verosímil: redacta la frase de modo que
  no lo necesite, o deja constancia de que consta en autos. Un dato inventado
  en un resultando se firma.
- EL TRIBUNAL ES EL DE ARRIBA, no otro. La competencia se funda en
  {_ta_r.cadena_competencia(_tipo)}, con el acuerdo general que
  fije la jurisdicción territorial de ese tribunal —si no sabes cuál es, no
  cites acuerdo alguno—.
- FRASE de unas 35 palabras, subordinada. Voz impersonal: «se estima», «este
  Tribunal Colegiado considera». Nunca primera persona del singular.
- Sin Markdown, sin viñetas.

Devuelve SÓLO este JSON:
{{"apertura": "<ciudad y fórmula de resolución del tribunal; la FECHA DE LA SESIÓN se deja como ___ porque aún no ocurre>",
  "visto": "<{_molde_visto}. SIN repetir el rótulo V I S T O / V I S T O S, que lo pone el compositor>",
  "resultandos": [
{_resultandos}
  ],
  "competencia": "<por qué este tribunal es competente, con su fundamento>",
  "existencia": "<la existencia del acto reclamado, acreditada con el informe justificado y los autos>",
  "procedencia": "<que el juicio es procedente y no se advierte causa de improcedencia, o cuál>"}}"""


async def redactar_estructura(cliente, datos: dict) -> Estructura:
    kw = dict(model=MODELO_ESTRUCTURA,
              max_completion_tokens=MAX_TOKENS_ESTRUCTURA,
              messages=[{"role": "user", "content": prompt_estructura(datos)}])
    if ESFUERZO_ESTRUCTURA:
        kw["reasoning_effort"] = ESFUERZO_ESTRUCTURA
    import llamada_modelo as _lm
    r = await _lm.crear(cliente, **kw)
    crudo = (r.choices[0].message.content or "").strip()
    m = _RX_JSON.search(crudo)
    if not m:
        return Estructura(avisos=[
            "El motor no devolvió las partes estructurales "
            f"({'sin texto' if not crudo else 'JSON ilegible'}). El documento "
            "sale sin resultandos ni competencia: revísalo antes de firmar."])
    try:
        d = json.loads(m.group(0))
    except Exception as e:
        return Estructura(avisos=[f"El JSON de la estructura no se pudo leer: {e}"])
    return Estructura(
        apertura=str(d.get("apertura", "")),
        visto=str(d.get("visto", "")),
        resultandos=[{"titulo": str(x.get("titulo", "")),
                      "texto": str(x.get("texto", ""))}
                     for x in (d.get("resultandos") or [])],
        competencia=str(d.get("competencia", "")),
        existencia=str(d.get("existencia", "")),
        procedencia=str(d.get("procedencia", "")))


# ═══════════════════════════════════════════════════════════════════════════
# EL MARCO JURÍDICO — se compone, no se pide
# ═══════════════════════════════════════════════════════════════════════════
# Dos intentos por prompt fallaron: el material del bloque de
# constitucionalidad llegaba al estudio —6,400 caracteres con el artículo 4º y
# la Convención sobre los Derechos del Niño— y el modelo no lo escribía. Se
# movió al 93% del prompt y siguió sin escribirlo.
#
# Lo que tiene que aparecer se compone. El marco pasa a ser un apartado propio
# del CONSIDERANDO, escrito por su propia llamada y colocado por el compositor
# ANTES del estudio. Sigue dependiendo del caso —si el acervo no devuelve nada
# constitucional ni convencional, el apartado no existe—, que es lo que David
# pidió: «no fijar un marco constitucional para todos los casos, sino sobre la
# solución en función del problema jurídico».

# Lo que se nombra tiene que estar en el material. En el ADC 380/2025 el
# estudio citó la Convención sobre los Derechos del Niño SEIS veces sin que la
# capa convencional se hubiera buscado siquiera: el modelo la escribió de
# memoria. Una cita que nadie puede comprobar es lo que este sistema existe
# para evitar, y da igual que la Convención exista: lo que no consta, no consta.
_RX_TRATADO = re.compile(
    r"(Convenci[óo]n\s+(?:sobre|Americana|Interamericana|de)[^.,;]{0,60}"
    r"|Pacto\s+(?:de\s+San\s+Jos[ée]|Internacional[^.,;]{0,40})"
    r"|Protocolo\s+de\s+San\s+Salvador)", re.I)
_RX_COIDH = re.compile(r"Corte\s+Interamericana|CoIDH|caso\s+[A-ZÁÉÍÓÚÑ][\w]+\s+Vs\.",
                       re.I)


def revisar_marco(marco_escrito: str, material_marco: str) -> list:
    """Lo que el marco nombra y el acervo no respalda."""
    fuera = []
    if not (marco_escrito or "").strip():
        return fuera
    mat = material_marco or ""
    nombrados = {m.group(0).strip() for m in _RX_TRATADO.finditer(marco_escrito)}
    sin_respaldo = [x for x in nombrados
                    if x.split()[0].lower() not in mat.lower()
                    or not any(p.lower() in mat.lower()
                               for p in x.split()[:4] if len(p) > 4)]
    if sin_respaldo:
        fuera.append(
            f"El marco nombra instrumentos que NO están en el material "
            f"recuperado: {sorted(sin_respaldo)[:4]}. El modelo los escribió de "
            f"memoria; compruébalos antes de firmar.")
    if _RX_COIDH.search(marco_escrito) and not _RX_COIDH.search(mat):
        fuera.append(
            "El marco invoca a la Corte Interamericana y el acervo no devolvió "
            "ni un fragmento suyo para este asunto. Sin la fuente no se cita.")
    return fuera


async def redactar_marco(cliente, material_marco: str, problemas: list,
                         es_recurso: bool = False,
                         tipo_asunto: str = "") -> str:
    """El marco jurídico, escrito. Devuelve texto vacío si no hay material."""
    if not (material_marco or "").strip():
        return ""
    # La bisagra de cierre —«…dar solución a los planteamientos de la parte
    # quejosa»— estaba escrita a mano y duplicada literal en el prompt del
    # estudio: es una fórmula que el modelo copia palabra por palabra, así que
    # metía «la parte quejosa» en el punto de bisagra de los cuatro tipos.
    _t = tipo_asunto or ("amparo_revision" if es_recurso else "amparo_directo")
    _voc_m = _ta.vocabulario_de(_t)
    _parte_prosa = _voc_m["parte"]
    q = _voc_m["combate"]
    lista = "\n".join(
        f"- {p.get('pregunta','') if isinstance(p, dict) else str(p)}"
        for p in (problemas or []))
    prompt = f"""Eres el secretario de un Tribunal Colegiado y escribes el apartado
de MARCO JURÍDICO de una sentencia de amparo: la premisa mayor con la que
después se resolverán los planteamientos.

LOS PROBLEMAS QUE HAY QUE RESOLVER
{lista}

MATERIAL RECUPERADO DEL BLOQUE DE CONSTITUCIONALIDAD Y DE LA LEY QUE RIGE EL ACTO
{material_marco[:12000]}

CÓMO SE ESCRIBE, medido sobre los engroses de este tribunal:
- ARRANCA POR LA FIGURA JURÍDICA discutida —los alimentos, la convivencia, la
  acción—, NO por los derechos humanos en abstracto.
- LA CONSTITUCIÓN SE PARAFRASEA, no se transcribe: «el artículo 4º de la
  Constitución reconoce el derecho de la niñez a…». Nómbrala expresamente, con
  su número de artículo.
- Y NOMBRA TODOS LOS ARTÍCULOS CONSTITUCIONALES QUE TE DIERON. Están arriba
  porque el asunto los tocó; el que no escribas queda sin premisa. Si uno de
  verdad no viene al caso, DILO en una frase —«el artículo X no rige aquí
  porque…»— en vez de callarlo: el silencio no se distingue del olvido.
- EL PRECEPTO SECUNDARIO decisivo —el de la ley que rige el acto, sea federal o
  local; dilo según su fuero, nunca «local» si la ley es federal— SÍ se transcribe, entre comillas y
  con su número al frente.
- LA FUENTE CONVENCIONAL —Convención sobre los Derechos del Niño, Convención
  Americana— y los criterios de la CORTE INTERAMERICANA entran SÓLO si el
  problema los exige. Cuando entran, se dice qué obligación imponen, no que
  existen.
- SI ARRIBA HAY MATERIAL DE LA CORTE INTERAMERICANA, ÚSALO. Está ahí porque el
  acervo lo encontró para ESTOS problemas, y desaprovecharlo deja el marco a
  medias. Se cita por el caso y el párrafo que trae la ficha —«Caso X Vs. Y,
  párr. N»— y se dice qué estándar fija, no que existe. Si de veras no viene al
  caso, no lo pongas; pero entonces tampoco cites la Corte de memoria.
- NO INVENTES NADA que no esté en el material de arriba. Ni un artículo, ni un
  caso, ni un párrafo de cuadernillo.
- EXTENSIÓN: entre 300 y 600 palabras. Frase de unas 35 palabras, subordinada,
  voz impersonal. Sin Markdown ni viñetas.
- CIERRA con la bisagra que devuelve al expediente: «Con ese marco jurídico, es
  posible dar solución a los planteamientos de {_parte_prosa}.»

Devuelve SÓLO el texto del apartado, en párrafos separados por una línea en
blanco. Sin rótulo ni encabezado: el documento se lo pone."""
    kw = dict(model=MODELO_ESTRUCTURA,
              max_completion_tokens=MAX_TOKENS_ESTRUCTURA,
              messages=[{"role": "user", "content": prompt}])
    if ESFUERZO_ESTRUCTURA:
        kw["reasoning_effort"] = ESFUERZO_ESTRUCTURA
    import llamada_modelo as _lm
    r = await _lm.crear(cliente, **kw)
    return (r.choices[0].message.content or "").strip()


# ═══════════════════════════════════════════════════════════════════════════
# La composición
# ═══════════════════════════════════════════════════════════════════════════

_ORDINALES = ("PRIMERO", "SEGUNDO", "TERCERO", "CUARTO", "QUINTO", "SEXTO",
              "SÉPTIMO", "OCTAVO", "NOVENO", "DÉCIMO")

_AMPARA = "ampara y protege"
_NO_AMPARA = "no ampara ni protege"

# ═══════════════════════════════════════════════════════════════════════════
# EL RESOLUTIVO NO ES EL MISMO EN TODOS LOS ASUNTOS
# ═══════════════════════════════════════════════════════════════════════════
# Se generaron cinco asuntos reales del corpus del secretario y se compararon
# con sus engroses. Tres de los cinco salieron con un resolutivo JURÍDICAMENTE
# IMPOSIBLE, porque esta función escribía la fórmula del amparo directo pasara
# lo que pasara:
#
#   queja QA 143/2026    engrose: «ÚNICO. Es fundado el recurso de queja.»
#                        motor:   «La Justicia de la Unión ampara y protege…»
#   revisión fiscal 6/25 engrose: «ÚNICO. Se confirma la sentencia de cuatro de
#                                  octubre…, dictada en el expediente 293/24…»
#                        motor:   «La Justicia de la Unión ampara y protege…»
#
# Una QUEJA no ampara: se declara fundada o infundada. Una REVISIÓN no ampara:
# confirma, revoca o modifica la sentencia recurrida. Sólo el amparo directo
# —y el resolutivo de fondo de una revisión que ampara— usan la fórmula de la
# Justicia de la Unión. Escribirla en una queja no es un defecto de estilo: es
# una resolución que no existe en derecho.
#
# Las fórmulas salen LITERALES de los engroses del propio tribunal, no de mi
# idea de cómo se redactan.
RESOLUTIVO = {
    "queja": {
        "punto": "Es {calificacion} el recurso de queja.",
        "calif": ("fundado", "infundado"),
        "notif": ("Notifíquese; publíquese y anótese en el libro de control de "
                  "este tribunal, hágase la captura correspondiente en el "
                  "Sistema Integral de Seguimiento de Expedientes, envíese "
                  "testimonio de esta resolución al juzgado de origen y, en su "
                  "oportunidad archívese como asunto concluido."),
    },
    "revision_fiscal": {
        # LA FÓRMULA MEDIDA, no la mía. `banco_formulas_medidas.json` la cuenta
        # en 16 de 28 revisiones fiscales del tribunal: «Se confirma la
        # sentencia DE {fecha}, dictada EN EL EXPEDIENTE {expediente}, por la
        # Sala…». Aquí se escribía «la sentencia recurrida, dictada por la
        # Sala», que no dice CUÁL sentencia se revoca —y en un tribunal donde
        # la misma Sala dicta cientos, identificarla no es un adorno—.
        #
        # Los dos datos se leen: el expediente con `fase_origen.numero_de`, que
        # acierta 5 de 5 sobre los engroses reales, y la fecha con `fecha_de`,
        # que ahora calla cuando duda. Lo que no se pudo leer sale en hueco, a
        # la vista, con su aviso.
        "punto": ("Se {calificacion} la sentencia de {fecha_sentencia}, dictada "
                  "en el expediente {expediente_origen}, por {responsable}."),
        "calif": ("revoca", "confirma"),
        "notif": ("Notifíquese; publíquese y anótese en el Libro de control de "
                  "este Tribunal, hágase la captura correspondiente en el "
                  "Sistema Integral de Seguimiento de Expedientes, con "
                  "testimonio de esta resolución vuelvan los autos a su lugar "
                  "de origen y, en su oportunidad archívese como asunto "
                  "concluido."),
    },
    "amparo_revision": {
        "punto": "Se {calificacion} la sentencia recurrida, dictada por {responsable}.",
        "calif": ("revoca", "confirma"),
        "notif": ("Notifíquese; publíquese y anótese en el libro de control de "
                  "este tribunal, hágase la captura correspondiente en el "
                  "Sistema Integral de Seguimiento de Expedientes, con "
                  "testimonio de esta resolución vuelvan los autos a su lugar "
                  "de origen y, en su oportunidad archívese como asunto "
                  "concluido."),
    },
    "amparo_directo": {
        "punto": None,          # lleva la fórmula de la Justicia de la Unión
        "notif": ("Notifíquese; publíquese y anótese en el libro de control de "
                  "este tribunal, hágase la captura correspondiente en el "
                  "Sistema Integral de Seguimiento de Expedientes, con "
                  "testimonio de esta resolución vuelvan los autos a su lugar "
                  "de origen y, en su oportunidad archívese como asunto "
                  "concluido."),
    },
}


# LO QUE TECLEA EL SECRETARIO TAMBIÉN SE COMPONE. En el proyecto 382/2024 el
# resolutivo salió diciendo «contra el acto que reclamó de la Junta especial 50
# de la federal de arbitraje en el estado de querétaro.,, precisado en el primer
# resultando». Tres defectos en una línea, y los tres del mismo origen: el campo
# se copiaba verbatim.
#
#   · el punto final que escribió el usuario, seguido de la coma de la
#     plantilla, da «.,» —y con la segunda coma de la frase, «.,,»—;
#   · las minúsculas, que en un resolutivo se leen como descuido;
#   · y el nombre convive en el mismo documento con la grafía correcta que el
#     modelo sacó del laudo, «Junta Especial Número Cincuenta de la Federal de
#     Conciliación y Arbitraje», así que el documento se contradice a sí mismo.
#
# No se cambia lo que el secretario escribió —eso es suyo y puede tener razones
# para nombrarla así—: se le quita la puntuación final y se le arreglan las
# mayúsculas si vino todo en minúsculas. Nada más.
_CONECTIVAS = {"de", "del", "la", "las", "el", "los", "y", "en", "e", "al",
               # LOS POSESIVOS Y LAS PREPOSICIONES TAMPOCO (revisión AR, 3-oct-2026;
               # AR 222/2025: «actuario de su adscripción» salía en el punto del
               # amparo «el Actuario de Su Adscripción», y «secretario ejecutor
               # adscrito a la dirección…», «Adscrito A la Dirección»).
               "a", "su", "sus", "o", "u", "por", "con", "para"}
# Lo que sigue a un posesivo («de su adscripción») y el participio «adscrito»
# no son parte del nombre del órgano: van en minúscula.
_RX_PARTICIPIO_ADSCRITO = re.compile(r"^adscrit[oa]s?$", re.I)


def _sin_articulo(x: str) -> str:
    """«el Juzgado Segundo» → «Juzgado Segundo». Lo pone la plantilla.

    EL CONTRATO ESTABA ESCRITO Y NO IMPLEMENTADO: el comentario de `_datos_bk`
    dice «se entrega el nombre limpio y la plantilla lo enmarca», y las
    fórmulas del banco dicen «dictado por el {responsable}». Pero nada quitaba
    el artículo, así que en cuanto el secretario teclea «el Juzgado Segundo de
    Distrito» en el formulario —que es como se dice— sale «por el el Juzgado».
    No pasaba con la autoridad LEÍDA del acto, que viene sin artículo; pasa con
    la que se escribe a mano, que es la que existe para corregir la leída.
    """
    return re.sub(r"^\s*(?:el|la|los|las)\s+", "", x or "", flags=re.I)


def _normalizar_autoridad(nombre: str) -> str:
    n = " ".join((nombre or "").split()).strip(" \t.,;:")
    if not n:
        return ""
    # Si trae mayúsculas propias, se respeta tal cual: el secretario sabe cómo
    # se llama la autoridad de su expediente mejor que yo.
    if any(c.isupper() for c in n[1:]):
        return n
    # LOS NÚMEROS ROMANOS NO SE CAPITALIZAN. `"II".capitalize()` devuelve «Ii»,
    # y el resolutivo salió diciendo «la Sala Regional del Centro Ii». Se
    # reconocen y se dejan como están.
    _romano = re.compile(r"^[IVXLCDM]{1,7}$")
    partes = []
    tras_posesivo = False
    for i, w in enumerate(n.split()):
        limpio = w.strip(".,;:()")
        if _romano.match(limpio.upper()) and limpio.upper() == limpio:
            partes.append(w)                       # ya viene en versales
        elif _romano.match(limpio.upper()) and len(limpio) > 1:
            partes.append(w.upper())               # «ii» → «II»
        elif i and (w.lower() in _CONECTIVAS or tras_posesivo
                    or _RX_PARTICIPIO_ADSCRITO.match(limpio)):
            partes.append(w)
        else:
            partes.append(w.capitalize())
        tras_posesivo = limpio.lower() in ("su", "sus")
    return " ".join(partes)


# CONTRA QUÉ SE RECURRE, en las palabras del oficio. El banco lo midió: «del
# acuerdo que desechó la demanda de amparo», «de un auto dictado en un juicio de
# amparo indirecto». Si no se puede decir con precisión, se dice lo genérico
# —que es cierto— en vez de dejar un hueco: un considerando de competencia con
# un agujero en mitad de la frase no se puede leer en sesión.
_GENERICO_ACTO = {
    "queja": "del auto recurrido",
    "amparo_revision": "de la sentencia recurrida",
    "revision_fiscal": "de la sentencia recurrida",
    "amparo_directo": "de la sentencia reclamada",
}


def _descripcion_del_acto(datos: dict, tipo: str) -> str:
    d = " ".join(str(datos.get("descripcion_acto") or "").split()).strip(" .,;")
    if d:
        return d if d.lower().startswith(("del ", "de ", "de la ")) else f"del {d}"
    return _GENERICO_ACTO.get(str(tipo or "").strip().lower(), "del acto recurrido")


def _de_la(nombre: str) -> str:
    """«de la Sala Regional…», «del Tribunal Unitario…».

    El resolutivo decía «contra el acto que reclamó de el Tribunal Unitario
    Agrario», porque la plantilla ponía «de » delante de lo que `_con_articulo`
    devolvía con su artículo. «De el» no es español: se contrae.
    """
    n = _con_articulo(nombre)
    if not n:
        return ""
    if n.lower().startswith("el "):
        return "del " + n[3:]
    return "de " + n


# EL ARTÍCULO CONTRAÍDO. «contra el acto reclamado a el Director» no es
# español, y sale en cuanto una plantilla escribe la preposición y otra función
# pone el artículo: ninguna de las dos ve a la otra. Se arregla al final, sobre
# el texto ya armado, que es donde las dos se juntan.
def _contraer(t: str) -> str:
    t = re.sub(r"\ba\s+el\b(?!\s+que\b)", "al", t)
    return re.sub(r"\bde\s+el\b(?!\s+que\b)", "del", t)


def _rec_es_autoridad(nombre: str) -> bool:
    n = (nombre or "").lower()
    return any(k in n for k in (
        "titular", "director", "directora", "unidad de", "secretaría", "secretaria de",
        "juzgado", "tribunal", "sala ", "ayuntamiento", "instituto", "comisión", "comision",
        "fiscal", "procurad", "presidente municipal", "gobernador", "congreso",
        "servicio de administración", "subsecretar", "jefe de", "administrador", "autoridad"))


# LOS CARGOS EN FEMENINO Y LOS ÓRGANOS QUE NO CABÍAN EN LA REGLA (3-oct-2026,
# tercera ronda). Banco de oráculo: «el Coordinadora de Recursos Humanos» (AD
# 128/2025), «el Jefa de la Unidad Jurídica», «el Subdirectora de Afiliación»
# (RF 2, 7 y 26/2025), «el Legislatura del Estado de Querétaro» (AR 201/2025).
# El cargo lo da el papel —«Jefa», «Directora»— y su artículo concuerda con
# ESA palabra, no con quien lo ocupa: no es adivinar el género de una persona.
_FEMENINAS_CARGO_ORGANO = {
    "jefa", "subjefa", "delegada", "subdelegada", "encargada", "presidenta",
    "comisionada", "consejera", "contralora", "tesorera", "subtesorera",
    "subsecretaria", "gerenta", "ministra", "síndica", "sindica",
    "legislatura", "cámara", "camara", "jefatura", "subjefatura", "oficina",
    "sindicatura", "gubernatura", "regiduría", "regiduria", "alcaldía", "alcaldia",
    # LOS TRATAMIENTOS EN FEMENINO (3-oct-2026, cuarta ronda; AR 239/2025 del
    # banco): «el licenciada Bertha Martínez Vega». El tratamiento lo escribe
    # el papel; su artículo concuerda con esa palabra, no con el nombre.
    "licenciada", "maestra", "mtra.", "doctora", "dra.",
}
# LOS NOMBRES DE PILA QUE ACABAN COMO UN CARGO. La regla -dora/-tora/-sora es
# de cargos («Coordinadora», «Procuradora», «Inspectora») y esta función
# recibe órganos, no personas; pero si un nombre de persona llegara aquí, su
# artículo NO se decidiría por el nombre de pila: «Isadora», «Teodora»,
# «Salvadora», «Pastora» quedan fuera de la regla.
_NOMBRES_DE_PILA_EN_ORA = {"dora", "isadora", "teodora", "heliodora", "salvadora", "amadora",
                           "pastora", "nestora", "melchora", "victora", "auxiliadora"}
# Y LA REGLA SÓLO VALE PARA LO QUE TIENE FORMA DE CARGO: la palabra sola o con
# su complemento al lado («Coordinadora de…», «Directora General»,
# «Subdirectora Jurídica»).
_COMPLEMENTO_DE_CARGO = {"de", "del", "en", "general", "jurídica", "juridica", "regional",
                         "estatal", "municipal", "federal", "ejecutiva", "adjunta",
                         "técnica", "tecnica", "administrativa", "fiscal", "titular",
                         "auxiliar", "especial", "local", "interina", "suplente"}
# LOS PREFIJOS TEMPORALES (RF 49/2025: «dictada por el Actual Sala Regional en
# Querétaro…, localizado»). «actual», «entonces», «hoy», «ahora» y «otrora»
# califican al órgano, no lo nombran: van en minúscula, detrás del artículo,
# y el artículo lo decide el sustantivo que sigue —«la actual Sala»—.
_RX_PREFIJO_TEMPORAL = re.compile(r"^(actual|entonces|hoy|ahora|otrora)\s+(?=\S)", re.I)


def _es_cargo_femenino_en_ora(primera: str, palabras: list) -> bool:
    """«Coordinadora de Recursos Humanos», «Directora General», «Inspectora»."""
    if not re.search(r"(?:d|t|s)ora$", primera) or primera in _NOMBRES_DE_PILA_EN_ORA:
        return False
    if len(palabras) == 1:
        return True
    return any(w.lower().strip(".,;:") in _COMPLEMENTO_DE_CARGO for w in palabras[1:3])


def _con_articulo(nombre: str) -> str:
    """«Primera Sala Civil…» → «la Primera Sala Civil…».

    Sin esto el resolutivo dice «reclamó de Primera Sala Civil», que no es
    español. El artículo se elige por la primera palabra, y si ya viene con él
    no se duplica. Un prefijo temporal («actual», «entonces», «hoy») va en
    minúscula detrás del artículo y no lo decide: «la actual Sala Regional».
    """
    n = _normalizar_autoridad(nombre)
    if not n:
        return ""
    # AL HUECO NO SE LE PONE ARTÍCULO (3-oct-2026, cuarta ronda, E10): no se
    # sabe qué órgano es, y «dictada por el *********» afirma su género.
    if not n.strip("*"):
        return n
    if re.match(r"^(?:el|la|los|las)\s", n, re.I):
        return n
    _pre = ""
    _m_pre = _RX_PREFIJO_TEMPORAL.match(n)
    if _m_pre:
        _pre = _m_pre.group(1).lower() + " "
        n = n[_m_pre.end():]
    primera = n.split()[0].lower()
    if primera in _FEMENINAS_CARGO_ORGANO or _es_cargo_femenino_en_ora(primera, n.split()):
        return f"la {_pre}{n}"
    femeninas = ("sala", "junta", "primera", "segunda", "tercera", "cuarta",
                 "quinta", "sexta", "séptima", "octava", "novena", "décima",
                 "autoridad", "comisión", "procuraduría", "secretaría",
                 "dirección", "delegación", "subdelegación",
                 # LAS PERSONAS JUZGADORAS FALTABAN. El considerando de
                 # existencia del 410/2026 salió con «el Jueza de Distrito del
                 # Juzgado Quinto»: la lista sólo traía nombres de órganos y el
                 # informe justificado lo rinde una persona, con su cargo en
                 # femenino cuando corresponde. Escribir «el Jueza» en un
                 # proyecto es de las erratas que se leen a la primera.
                 # «TITULAR» VA CON «EL» (3-oct-2026): es cargo en masculino
                 # genérico —el corpus del circuito escribe «el Titular de» 59
                 # veces contra 12 «la Titular de»—, y estaba aquí con las
                 # femeninas: el compositor de resultandos escribía «el Titular
                 # de la Jefatura» y la legitimación de este archivo «la
                 # Titular», en el mismo proyecto.
                 "jueza", "magistrada", "presidenta", "actuaria",
                 "secretaria", "encargada", "administradora", "recaudadora",
                 "asamblea", "agencia", "fiscalía", "oficialía", "notaría",
                 "tesorería", "coordinación", "administración", "unidad")
    if primera in femeninas:
        return f"la {_pre}{n}"
    # LO QUE NO CABE EN UNA LISTA. En español son femeninos casi sin excepción
    # los acabados en -ción, -sión, -dad, -tad y -ía, y esos sufijos abundan en
    # los nombres de órganos —«Recaudación», «Universidad», «Contraloría»—. La
    # lista nunca los va a agotar; la regla sí los cubre.
    #
    # No se usa el simple «acaba en -a», que se lleva por delante «Sistema»,
    # «Programa» y «Problema», todos masculinos.
    if re.search(r"(?:ci[óo]n|si[óo]n|dad|tad|[íi]a)$", primera):
        return f"la {_pre}{n}"
    return f"el {_pre}{n}"


HUECO = "*********"

# EL PLURAL SALE DEL CATÁLOGO. Estaba escrito a mano aquí y en fase6_estudio, y
# `_calificacion_plural` descarta lo que no esté en el diccionario: una
# calificación nueva desaparecía del encabezado del estudio sin avisar.
import tipos_asunto as _ta_pl
_PLURAL = {k: v[0] for k, v in _ta_pl.CALIFICACIONES.items()}
_PLURAL_VIEJO = {"fundado": "fundados", "infundado": "infundados",
           "inoperante": "inoperantes", "ineficaz": "ineficaces",
           # INNECESARIO NO ES UNA CALIFICACIÓN DEL PLANTEAMIENTO. No se dice
           # que sea infundado —eso sería contestarlo— sino que no hace falta
           # entrar: queda sin materia porque el principal ya resolvió el
           # asunto. Va aquí para que la calificativa del rótulo lo diga.
           "innecesario": "innecesarios de estudiar"}


def _calificacion_plural(cs: list) -> str:
    """«fundados», «en parte fundados y en parte inoperantes»…

    La calificativa va PEGADA al rótulo del Estudio en 17 de 26 engroses:
    «SEXTO. Estudio. Los conceptos de violación son infundados.»
    """
    limpios = [c for c in cs if c in _PLURAL]
    if not limpios:
        return ""
    unicos = []
    for c in limpios:
        if c not in unicos:
            unicos.append(c)
    if len(unicos) == 1:
        return _PLURAL[unicos[0]]
    # «EN PARTE PARCIALMENTE FUNDADOS» NO SE ESCRIBE. Las calificaciones que ya
    # llevan su propio matiz —«parcialmente fundados», «esencialmente
    # fundados», «fundados pero insuficientes»— no admiten el «en parte»
    # delante: se les antepone «unos» y «otros», que es como se dice.
    _matiz = lambda c: any(x in _PLURAL[c] for x in
                           ("parcialmente", "esencialmente", "sustancialmente",
                            "pero insuficientes", "sin materia"))
    if len(unicos) == 2:
        if _matiz(unicos[0]) or _matiz(unicos[1]):
            return f"unos {_PLURAL[unicos[0]]} y otros {_PLURAL[unicos[1]]}"
        return f"en parte {_PLURAL[unicos[0]]} y en parte {_PLURAL[unicos[1]]}"
    return ", ".join(_PLURAL[c] for c in unicos[:-1]) + f" y {_PLURAL[unicos[-1]]}"


# ═══ LAS MARCAS DE CITA DEL RESUMEN ════════════════════════════════════════
# Las fases 1-3 anotan de dónde sale cada afirmación del resumen del acto
# reclamado: «[[p.82 §3]]». En la plantilla se convertían en nota al pie; el
# generador nuevo no las tocaba y salían LITERALES en el cuerpo —quince en el
# proyecto que leyó David—. Una referencia visible entre corchetes dobles no es
# una cita rota por descuido: es el andamio del redactor asomando en el papel.
# UN CORCHETE DE CIERRE O DOS. El modelo escribió «[[p.38 §3-4; p.39 §1]» con
# uno solo y la marca sobrevivió entera en el papel. Exigir la forma perfecta
# de algo que escribe un modelo es garantizar que un día no case.
_MARCA_CITA = re.compile(r"\s*\[\[([^\[\]]{2,120})\]\]?")
_UNA_CITA = re.compile(
    r"p{1,2}\.?\s*(\d{1,4})(?:\s*[-–]\s*\d{1,4})?"
    r"(?:\s*§+\s*(\d{1,3}(?:\s*[-–]\s*\d{1,3})?))?", re.I)

# Tres por párrafo. Veintidós llamadas apiladas en el último punto es lo que
# pasa cuando el modelo devuelve el apartado entero sin saltos de línea.
MAX_CITAS_POR_PARRAFO = 3

# Y UN TOPE TOTAL. Medido contra los engroses reales: el ARC 448-2025 firmado
# lleva TRES notas al pie y el proyecto generado llevaba VEINTIOCHO. Una nota
# por cada afirmación del resumen no es rigor, es ruido: el secretario anota
# las que sostienen lo que se discute, no todas las que podría. Seis deja
# margen sobre su media sin convertir el pie en un segundo documento.
MAX_NOTAS_DEL_RESUMEN = 6


def _citas_de(marca: str) -> list:
    fuera = []
    for trozo in re.split(r"[;,]", marca):
        m = _UNA_CITA.search(trozo)
        if m:
            fuera.append((m.group(1), m.group(2) or ""))
    return fuera


def _texto_de_nota(pagina: str, parrafo: str) -> str:
    if not parrafo:
        return f"Cfr. página {pagina} de la sentencia reclamada."
    p = re.sub(r"\s*[-–]\s*", " a ", parrafo)
    return (f"Cfr. página {pagina}, párrafos {p}, de la sentencia reclamada."
            if " a " in p else
            f"Cfr. página {pagina}, párrafo {p}, de la sentencia reclamada.")


# ═══ EL ARTÍCULO CITADO, CON SU TEXTO AL PIE ═══════════════════════════════
# David: «cuando el redactor cita artículos (de cualquier fuente) debería citar
# su contenido textual a pie de página; eso incrementará dramáticamente el
# valor argumentativo del proyecto». Tiene razón y es barato: el precepto ya
# está en el acervo, palabra por palabra. Quien revisa deja de tener que ir a
# buscarlo, y quien firma ve de un vistazo si el artículo dice lo que se le
# atribuye —que es donde se cuelan los errores que nadie detecta—.
_RX_ARTICULO_CITADO = re.compile(
    r"art[íi]culos?\s+(\d{1,4})(?:\s*(?:bis|ter|qu[áa]ter))?"
    r"(?:[^.;]{0,90}?(c[óo]digo|ley|constituci[óo]n|reglamento)[^.;,]{0,60})?",
    re.I)


# Las palabras que no distinguen una ley de otra.
_VACIAS_LEY = {"de", "del", "la", "el", "los", "las", "y", "en", "para", "por",
               "sobre", "estado", "estados", "unidos", "nacional", "general"}
_RX_NOMBRA_LEY = re.compile(
    r"constituci[óo]n|constitucional|c[óo]digo|\bley\b|reglamento|convenci[óo]n|"
    r"pacto|tratado|contrato\s+colectivo|condiciones\s+generales", re.I)


def _sin_tildes(x: str) -> str:
    import unicodedata
    x = unicodedata.normalize("NFKD", x or "")
    return "".join(c for c in x if not unicodedata.combining(c))


def _norma_del_texto(frag: str, num: str, normas: list):
    """El precepto del acervo que se está citando, si lo hay.

    Se exige que coincidan el NÚMERO y, cuando el texto nombra una ley, que la
    fuente comparta palabras con ella: el «artículo 296» del código civil de
    Querétaro y el «artículo 296» de otro cuerpo no son el mismo, y poner al
    pie el texto equivocado es peor que no poner nada.
    """
    ley = _sin_tildes((frag or "").lower())
    # ¿La cita nombra una ley, o dice «el artículo 17» a secas?
    nombra_ley = bool(_RX_NOMBRA_LEY.search(frag or ""))
    mejor, puntos = None, -99
    for n in (normas or []):
        if str(n.get("articulo", "")).strip() != str(num):
            continue
        # El acervo llama al campo `cuerpo_legal`; sólo algunas fuentes usan
        # `fuente`. Leer una sola de las dos daba CERO coincidencias y ninguna
        # nota de artículo salía, sin que nada avisara.
        fuente = _sin_tildes(str(n.get("cuerpo_legal") or n.get("fuente") or "").lower())
        suyas = {w for w in re.findall(r"[a-z]{4,}", fuente) if w not in _VACIAS_LEY}
        acierta = len([w for w in suyas if w in ley])
        # SE PENALIZA LO QUE SOBRA, igual que al traer los artículos por número.
        # Contando sólo aciertos, «Ley Federal de Responsabilidad Patrimonial
        # del Estado» empata con cualquier cosa que diga «Estado».
        p = acierta - len(suyas - {w for w in suyas if w in ley})
        if p > puntos:
            mejor, puntos = n, p
    # Y LA REGLA QUE FALTABA, QUE ES LA QUE COSTÓ UN PROYECTO. Antes, si ninguna
    # ley coincidía, esta función se quedaba con la PRIMERA norma que tuviera
    # ese número —`mejor is None and p == 0`—, viniera de donde viniera. Así el
    # documento transcribió, DENTRO DE COMILLAS y presentándolo como el artículo
    # 17 de la Constitución, el artículo 17 de la Ley Federal de Responsabilidad
    # Patrimonial del Estado: «Las resoluciones que se dicten con motivo de las
    # reclamaciones deberán contener… relación de causalidad entre el
    # funcionamiento del servicio público…». La prosa del modelo era correcta;
    # lo que mentía era la transcripción que yo le pegaba debajo.
    #
    # Si la cita nombra una ley y ninguna norma del material es de esa ley, NO
    # SE TRANSCRIBE NADA. Un artículo sin su texto se queda sin nota al pie y
    # quien firma lo comprueba a mano; un artículo con el texto de otra ley se
    # firma sin comprobar, y eso es lo que no se perdona.
    # Y SIN NOMBRE DE LEY TAMPOCO SE ADIVINA. El prompt exige desde hace
    # semanas nombrar la ley en la misma frase que el número —«el artículo 296
    # del Código Civil del Estado de Querétaro», nunca «el 296» a secas—; si
    # aun así llega pelado, elegir por él es apostar. Se queda sin nota y quien
    # firma lo comprueba, que es exactamente lo que la nota existe para
    # ahorrarle cuando SÍ se puede saber.
    if mejor is None or puntos < 1:
        return None
    return mejor


# CUÁNTO SE TRANSCRIBE DE UN ARTÍCULO. Medido en el proyecto de la queja civil
# 233/2025: el artículo 107 CONSTITUCIONAL salió transcrito ENTERO, 1,924
# palabras, con sus dieciocho fracciones y sus incisos, cuando lo que se
# discutía era una garantía de suspensión. Eso solo se llevaba una quinta parte
# del documento, disparaba la medida de transcripción por encima del engrose y
# generaba media docena de «pasajes duplicados» que eran trozos del mismo
# artículo repetidos.
#
# El secretario no hace eso: transcribe «en la parte conducente». Y el corpus lo
# dice con esas palabras —está en la dispensa de los cinco engroses—.
MAX_PALABRAS_PRECEPTO = 180

# A partir de aquí, la tesis se lee al pie. Ochenta palabras son unos cinco
# renglones: lo que cabe sin romper la lectura del razonamiento.
MAX_PALABRAS_TESIS_CUERPO = 80


def _en_lo_conducente(cuerpo: str, fraccion="") -> str:
    """El artículo, o su parte conducente si es largo.

    Primero se intenta quedarse con la FRACCIÓN que el párrafo citó, que es lo
    que el secretario haría. Si no consta cuál, se corta en frontera de frase y
    se dice «en lo conducente», que es como se anuncia una transcripción
    parcial: fingir que es íntegra cuando no lo es sería peor que cortarla.
    """
    pal = cuerpo.split()
    if len(pal) <= MAX_PALABRAS_PRECEPTO:
        return cuerpo
    # UNA O VARIAS. Cuando el párrafo cita más de una fracción del mismo
    # artículo hay que llegar a la ÚLTIMA: quedarse en la primera deja sin
    # respaldo la mitad del razonamiento. Se ordenan por dónde aparecen en el
    # texto, no por su número romano, porque lo que importa es hasta dónde hay
    # que leer.
    fracs = [fraccion] if isinstance(fraccion, str) else list(fraccion or [])
    fracs = [f for f in fracs if f]
    if fracs:
        _pos = {}
        for f in fracs:
            _m = re.search(rf"(?:^|[;.]\s*){re.escape(f)}\.\s", cuerpo)
            if _m:
                _pos[f] = _m.start()
        if _pos:
            # DE LA PRIMERA A LA ÚLTIMA, no sólo la última. Escrito como
            # estaba, «fracciones VI y VII» saltaba a la VII y se comía la VI
            # —la que el estudio razonaba primero—. Se abarca el tramo: se
            # empieza en la primera citada y se termina al acabar la última.
            fraccion = min(_pos, key=lambda f: _pos[f])
            _hasta = max(_pos, key=lambda f: _pos[f])
        else:
            fraccion = _hasta = ""
    else:
        fraccion = _hasta = ""
    if fraccion:
        # «IV.» o «fracción IV» dentro del texto del artículo.
        m = re.search(rf"(?:^|[;.]\s*){re.escape(fraccion)}\.\s", cuerpo)
        if m:
            resto = cuerpo[m.start():]
            # El final se busca DESPUÉS de la última fracción citada, para que
            # el tramo las contenga todas.
            _desde = 3
            if _hasta and _hasta != fraccion:
                _mu = re.search(rf"(?:^|[;.]\s*){re.escape(_hasta)}\.\s", resto)
                if _mu:
                    _desde = _mu.start() + 3
            fin = re.search(r"[;.]\s+[IVXLC]+\.\s", resto[_desde:])
            trozo = resto[:fin.start() + _desde] if fin else resto
            if 10 <= len(trozo.split()) <= MAX_PALABRAS_PRECEPTO * 2:
                # EL RÓTULO SE CORTABA EN LA ABREVIATURA. `split(".")[0]`
                # sobre «Art. 104.- Los Tribunales…» devuelve «Art», y el
                # precepto salía encabezado por «Art. […]», que no dice qué
                # artículo es. Se toma el rótulo entero, con su número.
                _r = re.match(r"\s*(Art[íi]culos?|Art)\.?\s*(\d{1,3}\s*"
                              r"(?:bis|ter)?)", cuerpo, re.I)
                cab = (f"Artículo {_r.group(2).strip()}" if _r
                       else cuerpo.split(".")[0])
                # EL PUNTO Y COMA VIAJABA PEGADO AL TROZO y salía «[…] ; III.
                # De los recursos…», con el espacio delante del signo. Se
                # quitan los signos de puntuación con que empieza el corte:
                # pertenecen a la frase anterior, que es la que se elidió.
                trozo = trozo.strip().lstrip(".;,: ")
                # Y EL RÓTULO NO SE ESCRIBE DOS VECES. El acervo guarda el
                # texto empezando por «Art. 104.-», así que anteponerle el
                # encabezado producía «Artículo 104. Art. […]».
                if re.match(r"^Art[íi]?c?u?l?o?\.?\s*\d", trozo):
                    return f"[…] {trozo}"
                return f"{cab}. […] {trozo}"
    corte = " ".join(pal[:MAX_PALABRAS_PRECEPTO])
    ult = max(corte.rfind(". "), corte.rfind("; "))
    if ult > len(corte) * 0.5:
        corte = corte[:ult + 1]
    # UN ROMANO SUELTO AL FINAL no es una fracción: es media fracción cortada.
    corte = re.sub(r"\s+[IVXLC]{1,6}\.?\s*$", "", corte.rstrip(" ;,."))
    return corte.rstrip(" ;,.") + " […]"


_RX_ROMANOS = r"[IVXLC]{1,6}"


def _fracciones_citadas(texto: str, num) -> list:
    """Las fracciones de ESE artículo que el párrafo cita, en cualquier orden.

    Un secretario escribe las dos formas sin pensarlo —«artículo 79, fracción
    V» y «la fracción V del artículo 79»— y a veces varias de golpe
    —«fracciones VI y VII»—. Reconocer sólo la primera forma dejó el artículo
    79 cortado en la fracción IV en un proyecto cuyo razonamiento giraba
    alrededor de la V.

    Se devuelven todas: transcribir de más es un párrafo largo; transcribir de
    menos es dejar sin respaldo la razón que se acaba de dar.
    """
    t = " ".join(str(texto or "").split())
    n = re.escape(str(num))
    fuera, vistos = [], set()
    for rx in (
            # «artículo 79, fracción V» · «artículo 79, fracciones VI y VII»
            rf"art[íi]culo\s+{n}\s*,?\s*fracci[óo]n(?:es)?\s+"
            rf"((?:{_RX_ROMANOS}(?:\s*(?:,|y|e)\s*)?)+)",
            # «la fracción V del artículo 79» · «las fracciones VI y VII del 79»
            rf"fracci[óo]n(?:es)?\s+((?:{_RX_ROMANOS}(?:\s*(?:,|y|e)\s*)?)+)"
            rf"\s*(?:de|del)\s+(?:art[íi]culo\s+)?{n}\b"):
        for m in re.finditer(rx, t, re.I):
            for r in re.findall(_RX_ROMANOS, m.group(1).upper()):
                if r not in vistos:
                    vistos.add(r)
                    fuera.append(r)
    return fuera


# ═══════════════════════════════════════════════════════════════════════════
# LA VERJA: UN TEXTO NO SE TRANSCRIBE COMO EL ARTÍCULO QUE NO ES
# ═══════════════════════════════════════════════════════════════════════════
# LA CAUSA DE FONDO, y no estaba en el acervo sino aquí. Las tres puertas por
# las que el texto de un artículo entra a la sentencia —la transcripción en
# sangría, la nota al pie de la prosa y la del marco jurídico— hacían todas lo
# mismo:
#
#     cuerpo = re.sub(r"^ART[ÍI]CULO\s+\d+[^.]{0,12}\.?\s*", "", cuerpo)
#     pie    = f"«Artículo {num}. {cuerpo}» — {ley}"
#
# Es decir: BORRABAN la cabecera que traía el texto y escribían la del artículo
# que se había citado. Con eso, cualquier fallo de recuperación —el 50-A por el
# 50, el 16 transitorio de 1917 por el 16 constitucional— dejaba de ser un
# fallo visible y se convertía en una afirmación falsa, confiada y firmada: el
# documento decía «Artículo 50» sobre el texto del 50-A, y la única prueba de
# que no eran el mismo se había tachado una línea antes.
#
# El acervo guarda 50-A, 251 A y los transitorios bajo el mismo `articulo_num`,
# y eso no se arregla afinando la búsqueda: se arregla no fiándose de ella. La
# cabecera del texto es una PRUEBA, no ruido. Aquí se coteja en vez de
# borrarse, y el que no case no se transcribe.
#
# Los avisos del cotejo viajan por una lista de módulo, como los de
# `ensamblar_adelanto`: las tres puertas están demasiado adentro para
# devolverlos, y el secretario tiene que enterarse de qué precepto se quedó sin
# transcribir y por qué.
avisos_cotejo: list[str] = []

_RX_CABECERA_ART = re.compile(
    r"^\W{0,4}(?:art[íi]culos?|arts?\.?)\s*(\d{1,4})\s*[-–]?\s*"
    r"(?:(bis|ter|qu[áa]ter|qu[íi]nquies|[A-K])\b)?",
    re.I)
_RX_MIGAJA = re.compile(r"^\s*\[[^\]]{0,400}\]\s*")
# «A.- Las sentencias…»: así parte el acervo el «50-A», sin su número.
_RX_SUFIJO_SUELTO = re.compile(
    r"^\W{0,4}(?:bis|ter|qu[áa]ter|qu[íi]nquies|[A-K])\s*[.\-–)]", re.I)


def cotejar_articulo(texto: str, num) -> tuple:
    """¿Este texto ES el artículo `num`? Devuelve (veredicto, lo_que_anuncia).

    · «confirmado»   su cabecera anuncia ese mismo artículo, sin sufijo.
    · «sin_cabecera» no se anuncia: el acervo guarda muchos preceptos sin
                     repetir el rótulo y descartarlos dejaría sin transcripción
                     a leyes enteras. Se transcribe, pero no está probado.
    · «desmentido»   anuncia OTRO artículo, o el mismo con sufijo. No se
                     transcribe: es el caso que firmaba una mentira.
    """
    t = _RX_MIGAJA.sub("", " ".join(str(texto or "").split()))
    if not t:
        return "desmentido", ""
    if _RX_SUFIJO_SUELTO.match(t):
        return "desmentido", t[:12].strip()
    m = _RX_CABECERA_ART.match(t)
    if not m:
        return "sin_cabecera", ""
    dice = m.group(1) + (("-" + m.group(2).upper()) if m.group(2) else "")
    try:
        pedido = str(int(str(num).strip()))
    except (TypeError, ValueError):
        pedido = str(num).strip()
    if m.group(2) or m.group(1) != pedido:
        return "desmentido", dice
    return "confirmado", dice


def cuerpo_para_transcribir(texto: str, num, ley: str = "") -> str:
    """El texto listo para ir entre comillas, o «» si no es ese artículo.

    Quita la migaja del acervo y la cabecera —que ya se cotejó— para que no
    salga «Artículo 14. Art. 14.- A ninguna ley…». Cuando el cotejo desmiente,
    devuelve vacío y deja anotado por qué: la cita del artículo se queda sin
    transcripción, que es un proyecto incompleto y honrado en vez de uno
    completo y falso.
    """
    veredicto, dice = cotejar_articulo(texto, num)
    # NI UNA RESPUESTA DE CHAT, venga de donde venga. La verja miraba si el
    # texto decía ser OTRO artículo; no si era un artículo en absoluto. Así
    # entró al 2/2026 la negativa del buscador web como transcripción del
    # artículo 150: «El **artículo 150** que corresponde al… no puedo citarlo
    # textualmente». Aquí se cierra para las tres puertas a la vez.
    try:
        import texto_normativo as _tn
        _norma_ok, _por_que_no = _tn.es_texto_normativo(texto)
    except Exception:
        _norma_ok, _por_que_no = True, ""
    if not _norma_ok:
        aviso = (f"NO SE TRANSCRIBIÓ EL ARTÍCULO {num}"
                 + (f" de {ley}" if ley else "")
                 + f": el texto que llegó no es el de la norma ({_por_que_no}). "
                 + "El artículo sigue citado, pero sin transcripción: cópialo de "
                 + "su publicación oficial.")
        if aviso not in avisos_cotejo:
            avisos_cotejo.append(aviso)
        return ""
    if veredicto == "desmentido":
        aviso = (f"NO SE TRANSCRIBIÓ EL ARTÍCULO {num}"
                 + (f" de {ley}" if ley else "")
                 + ": el texto que el acervo devolvió se anuncia como "
                 + (f"«{dice}»" if dice else "otro precepto")
                 + ". El artículo sigue citado, pero sin su transcripción: "
                 + "búscalo y pégalo tú, o corrige la cita.")
        if aviso not in avisos_cotejo:
            avisos_cotejo.append(aviso)
        return ""
    cuerpo = _RX_MIGAJA.sub("", " ".join(str(texto or "").split()))
    # La cabecera ya hizo su trabajo —probar de quién es el texto— y estorba
    # dentro de la comilla, porque el compositor escribe la suya.
    # La comilla de apertura va DELANTE de la cabecera cuando el texto viene
    # transcrito de su fuente oficial: «Artículo 150. “Artículo 150. Son
    # atribuciones…». El recorte exigía que la cabecera empezara el texto y la
    # comilla se lo impedía, así que el rótulo salía dos veces.
    for _ in range(2):
        cuerpo = re.sub(
            r"^[\s«»\"'“”]*ART(?:[ÍI]CULOS?)?\.?\s*\d+[^.]{0,14}\.?\s*[-–]?\s*",
            "", cuerpo, flags=re.I)
    return cuerpo.strip(" \t«»\"'“”")


def escribir_precepto(doc, texto_articulo: str, ley: str, num: str,
                      fraccion=""):
    """El artículo transcrito, como lo hace el secretario.

    David: «Cuando citamos un artículo hay que hacerlo con interlineado uno y
    con sangría en todo el artículo… No se dice en el mismo párrafo, se abren
    dos puntos, se cita textualmente el artículo y luego se sigue con la
    redacción». Así queda:

        …conforme al artículo 296 del Código Civil del Estado de Querétaro,
        que dispone:
            «Artículo 296. Los alimentos han de ser…»        ← sangrado, a uno
        Como se advierte del precepto transcrito, …          ← sigue la prosa

    El precepto sale del acervo, palabra por palabra. Un artículo transcrito de
    memoria es el error que nadie detecta al revisar.
    """
    cuerpo = " ".join(str(texto_articulo or "").split())
    if not cuerpo:
        return None
    cuerpo = _en_lo_conducente(cuerpo, fraccion)
    # POR LA VERJA. Quita la migaja del acervo y la cabecera duplicada —que es
    # lo que antes hacía este bloque— pero sólo DESPUÉS de comprobar que esa
    # cabecera dice ser el artículo que se está citando. Si dice otra cosa,
    # devuelve vacío y aquí no se transcribe nada.
    cuerpo = cuerpo_para_transcribir(cuerpo, num, ley)
    if not cuerpo:
        return None
    q = doc.add_paragraph()
    r = q.add_run(f"«Artículo {num}. {limpiar_texto_web(cuerpo)}»")
    _fmt(q, sangria=False, tamano=TAMANO_CITA,
         interlineado=INTERLINEADO_CITA)
    q.paragraph_format.left_indent = SANGRIA_CITA
    q.paragraph_format.right_indent = Cm(0.5)
    q.paragraph_format.space_before = Pt(6)
    q.paragraph_format.space_after = Pt(6)
    q.paragraph_format.keep_with_next = False
    return q


# LA COLA DE LA LEY VA EN UN LOOKAHEAD: consumida, se tragaba la cita que
# viniera detrás dentro de sus 150 caracteres.
_RX_ARTICULOS_CITADOS_LISTA = re.compile(
    r"art[íi]culos?\s+(\d{1,4}(?:\s*(?:º|°|o\.|bis|ter|qu[áa]ter))?"
    r"(?:\s*(?:,|y|e)\s*\d{1,4}(?:\s*(?:º|°|o\.|bis|ter))?)*)"
    r"(?=([^.;]{0,90}?(c[óo]digo|ley|constituci[óo]n|reglamento)[^.;,]{0,60})?)", re.I)

# ═══ LA LEY NOMBRADA CON OTRAS PALABRAS (AR 631/2025, 28-sep-2026) ═════════
# «La regla que la recurrente identifica como artículo 49 de la legislación
# procesal civil del Estado de Querétaro…»: sin «código» ni «ley» en la cola,
# el artículo no tenía ley, `_norma_del_texto` no lo casaba y no bajaba al pie;
# y `preceptos_fuera`, por voces sueltas, lo mandaba a traer del CÓDIGO CIVIL
# si éste ya estaba en el material. Las perífrasis se llevan al nombre del
# código ANTES de casar, aquí y en `fase6_estudio.citas_de_articulos`. Sólo se
# lee: el texto del proyecto no se toca.
_PERIFRASIS_LEY = (
    (re.compile(r"\b(?:legislaci[óo]n|ley|c[óo]digo|ordenamiento|norma)\s+"
                r"(?:adjetiva|procesal)\s+civil(?:es)?\b", re.I),
     "Código de Procedimientos Civiles"),
    (re.compile(r"\b(?:legislaci[óo]n|ley|ordenamiento|norma|c[óo]digo)\s+sustantiva\s+civil\b"
                r"|\blegislaci[óo]n\s+civil\b", re.I),
     "Código Civil"),
    # «del Código Civil local»: el fuero sin el nombre del estado. Sin esto
    # `fase6_rag.fuero_de` no lo ve estatal y el precepto se buscaba primero
    # en el silo federal de la materia.
    (re.compile(r"(?<=[A-Za-zÁÉÍÓÚáéíóúñÑ])\s+(?:local|de\s+(?:la|esta)\s+entidad(?:\s+federativa)?)\b", re.I),
     " del Estado"),
)
# «el artículo 2294 del mismo ordenamiento», «del citado código», «de dicha ley».
_RX_ANAFORA_LEY = re.compile(
    r"^\s*,?\s*(?:de\s+la|del|de)\s+(?:mism[oa]|propi[oa]|citad[oa]|dich[oa]|es[ea]|"
    r"aludid[oa]|referid[oa]|invocad[oa])\s+(?:ordenamiento|c[óo]digo|ley|cuerpo\s+"
    r"(?:legal|normativo)|constituci[óo]n|legislaci[óo]n)\b"
    r"|^\s*,?\s*del\s+(?:ordenamiento|c[óo]digo)\s+(?:citado|en\s+cita|invocado|mencionado)\b",
    re.I)


def canonizar_ley(texto: str) -> str:
    """El texto con las perífrasis de ley llevadas al nombre del código."""
    t = texto or ""
    for rx, nombre in _PERIFRASIS_LEY:
        t = rx.sub(nombre, t)
    return t


def _iter_articulos(texto: str):
    """(número, fragmento) por cada artículo citado; «artículos 134 y 137 del
    Código Fiscal de la Federación» rinde dos, los dos con la ley. «Del mismo
    ordenamiento» hereda la ley de la cita anterior del mismo párrafo."""
    t = canonizar_ley(texto or "")
    ley_previa = ""
    for m in _RX_ARTICULOS_CITADOS_LISTA.finditer(t):
        cola = m.group(2) or ""
        frag = m.group(0) + cola
        if _RX_ANAFORA_LEY.match(t[m.end():]):
            frag = m.group(0) + (" " + ley_previa if ley_previa else "")
        elif m.group(3):
            ley_previa = cola[cola.lower().find(m.group(3).lower()):]
        for num in re.findall(r"\d{1,4}", m.group(1)):
            yield num, frag


def _preceptos_del_parrafo(texto: str, normas: list) -> list:
    """[(número, norma)] de los artículos que este párrafo cita y tenemos."""
    fuera, vistos = [], set()
    for num, frag in _iter_articulos(texto):
        if num in vistos:
            continue
        n = _norma_del_texto(frag, num, normas)
        if n and str(n.get("texto") or "").strip():
            vistos.add(num)
            fuera.append((num, n))
    return fuera


def notas_de_articulos(doc, p, texto: str, normas: list, notas: list) -> int:
    """Cuelga del párrafo el texto de los artículos que cita. Devuelve cuántos."""
    if not normas:
        return 0
    puestos = 0
    for num, frag in _iter_articulos(texto):
        if puestos >= MAX_ARTICULOS_POR_PARRAFO:
            break
        n = _norma_del_texto(frag, num, normas)
        if not n:
            continue
        cuerpo = " ".join(str(n.get("texto", "")).split())[:900]
        if not cuerpo:
            continue
        _ley = n.get("cuerpo_legal") or n.get("fuente") or ""
        # POR LA VERJA: se coteja la cabecera antes de quitarla. El que no sea
        # el artículo citado se queda sin nota, y el aviso lo dice.
        cuerpo = cuerpo_para_transcribir(cuerpo, num, _ley)
        if not cuerpo:
            continue
        pie = (f"«Artículo {num}. {limpiar_texto_web(cuerpo)}» — {_ley}".strip()
               + marca_de_origen(n))
        if pie in notas:
            continue
        notas.append(pie)
        _run_llamada(p, len(notas))
        puestos += 1
    return puestos


def marca_de_origen(n: dict) -> str:
    """La coleta que dice que este texto NO salió del acervo verificado.

    Vive suelta porque el .docx compone notas de artículo en DOS sitios
    —`notas_de_articulos` y el marco jurídico— y la primera vez sólo se marcó
    uno: el artículo 150 del Reglamento Interior del IMSS, traído de
    imss.gob.mx en la revisión fiscal 2/2026, salió al pie sin una palabra
    sobre su procedencia. Quien firma no puede distinguir a ojo un precepto
    cotejado contra la base de uno transcrito de un sitio.
    """
    if not (isinstance(n, dict) and n.get("de_internet")):
        return ""
    dom = str(n.get("dominio") or "").strip()
    return (f" · TEXTO TOMADO DE {dom or 'una fuente en línea'}, no del acervo "
            f"verificado: COTÉJALO antes de firmar")


# LAS MIGAS DEL BUSCADOR. Sonar devuelve el texto con sus referencias
# incrustadas —«…circunscripción territorial: [1] I. Vigilar…»— y esas
# muletillas no pueden acabar dentro de un precepto citado en una sentencia.
_RX_CITA_WEB = re.compile(r"\s*\[\d{1,2}\]")


def limpiar_texto_web(t: str) -> str:
    return _RX_CITA_WEB.sub("", t or "")


MAX_ARTICULOS_POR_PARRAFO = 1
MAX_NOTAS_DE_ARTICULOS = 8
# UNA LLAMADA POR PÁRRAFO NO ES UN ARTÍCULO POR PÁRRAFO (AR 631/2025). La
# regla de la llamada única (661f4bd: las notas 3 y 4 se leían «34») se
# cumplía cortando la lista a su primer artículo: «los artículos 2284 y 2294
# del Código Civil…» bajaba el 2284 y el 2294 —el que decide— no aparecía en
# ninguna parte; igual el 17 de «artículos 14 y 17 de la Constitución». Los
# artículos de un párrafo van JUNTOS en una sola nota, uno por renglón.
MAX_ARTICULOS_POR_NOTA = 4


def parrafo_con_citas(doc, texto: str, notas: list):
    """Escribe el párrafo y baja sus marcas a notas al pie."""
    citas = [c for m in _MARCA_CITA.finditer(texto) for c in _citas_de(m.group(1))]
    limpio = _MARCA_CITA.sub("", texto).strip()
    if not limpio:
        return None
    p = parrafo(doc, limpio)
    # UNA LLAMADA POR PÁRRAFO, Y NUNCA DOS PEGADAS. Dos referencias seguidas sin
    # texto en medio se leen como un solo número: las notas 3 y 4 aparecían como
    # «34». Lo vio David. Y la MISMA página se citaba cinco veces seguidas
    # porque el modelo repite la marca párrafo tras párrafo; una nota repetida
    # no aporta nada y ensucia el pie.
    if not citas:
        return p
    pagina, parr = citas[0]
    texto_nota = _texto_de_nota(pagina, parr)
    if texto_nota in notas:
        return p                       # ya está al pie: no se repite
    if len([x for x in notas if x.startswith("Cfr.")]) >= MAX_NOTAS_DEL_RESUMEN:
        return p
    notas.append(texto_nota)
    _run_llamada(p, len(notas))
    return p


def _subtitulo(doc, texto: str):
    """Subtítulo en negrita SIN ordinal, como los del Estudio.

    Medido: «Sentencia reclamada», «Conceptos de violación», «Solución»,
    «Conclusión». No llevan número: no son considerandos, son las partes de
    uno solo.
    """
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12)
    r = p.add_run(texto)
    r.bold = True
    _fmt(p, sangria=True)
    p.paragraph_format.keep_with_next = True
    return p


# ANDAMIO DEL MODELO QUE NO DEBE LLEGAR AL PAPEL. En la queja salió, dentro de
# una jurisprudencia, «la jurisprudencia 2a./J. 58/2010,[NOTA 2] emitida por la
# Segunda Sala…». Nadie le enseñó esa etiqueta: la inventó imitando una
# convención de nota al pie, y el documento YA lleva notas de verdad —las
# inserta el compositor con su XML— así que ese corchete es un marcador
# huérfano que el secretario tiene que borrar a mano.
#
# El filtro es DELIBERADAMENTE ESTRECHO. Un limpiador de corchetes a secas se
# llevaría «[sic]», «[…]» y los incisos «[a]» de una transcripción, que sí son
# del documento. Sólo caen las etiquetas editoriales con su palabra clave.
_RX_ANDAMIO = re.compile(
    r"\s*\[\s*(?:NOTA|NOTAS|FOOTNOTE|CITA|REF|PIE)\s*[:#]?\s*\d*\s*\]", re.I)


def sin_andamio(texto: str) -> str:
    """Quita los marcadores que el modelo escribe para sí mismo."""
    t = _RX_ANDAMIO.sub("", texto or "")
    # El corchete se come el espacio de delante; si iba pegado a una coma,
    # queda «58/2010, emitida», que es lo que debía decir.
    return re.sub(r"\s+([,.;:])", r"\1", t)


# ═══════════════════════════════════════════════════════════════════════════
# LA FRASE QUE SIGUE A UNA CITA SE QUEDA SIN SUJETO
# ═══════════════════════════════════════════════════════════════════════════
# David, sobre la revisión fiscal 91/2025: «nuevamente perdiste el diálogo en el
# tercer problema. El error es la falta de conector lógico… empiezas con
# minúscula y "confirma que la Sala…", cuando debería ser "La tesis en cita,…"
# o algo equivalente».
#
# Y tiene razón en la causa: el modelo escribe UNA SOLA FRASE —«Sirve de apoyo
# la jurisprudencia …, de rubro y texto siguientes: «RUBRO» confirma que la
# Sala debe atender los conceptos…»— y el compositor la parte en tres para
# meter el bloque de la cita. Lo que queda detrás empieza en minúscula y sin
# sujeto: es media oración suelta.
#
# NO SE PUEDE ARREGLAR EN EL PROMPT SOLO, porque el corte lo hace el documento
# y el modelo no sabe dónde va a caer. Se repone el sujeto aquí, que es donde
# se conoce la cita: «La jurisprudencia en cita confirma que…».
#
# CON FRENO: sólo si la cola empieza en minúscula —eso es la marca inequívoca
# de la frase partida— y no empieza por conjunción, donde anteponer un sujeto
# produciría «La tesis en cita y confirma que…». Si no encaja, no se toca.
# Lo que NO es el verbo de una oración continuada: restos de la ficha de la
# tesis y arranques preposicionales.
_RX_ARRANQUE_NOMINAL = re.compile(
    r"^(?:registro|p[áa]gina|tomo|libro|volumen|[ée]poca|tesis|jurisprudencia|"
    r"n[úu]mero|clave|gaceta|semanario|instancia|materia|localizaci[óo]n|"
    r"de|del|en|por|con|para|al?|sobre|desde|hasta|entre|sin|seg[úu]n|"
    r"cuyo|cuya|cuyos|cuyas)\b", re.I)

# «también» NO ATA: es un adverbio, y la media frase que arranca con él sigue
# necesitando sujeto. ADC 93/2026 v4: tras la cita de la 2010224 quedó
# «también confirma que la ampliación no es una actuación accesoria…», en
# minúscula y sin sujeto, porque «también» estaba en esta lista.
_RX_ARRANQUE_ATADO = re.compile(
    r"^(?:y|e|o|u|pero|sino|aunque|que|porque|pues|como|cuando|si|ni|as[íi]|"
    r"adem[áa]s|donde|mientras|seg[úu]n|salvo)\b", re.I)


# ── LA FICHA QUE SE QUEDA AL FRENTE DE LA COLA ──────────────────────────────
# David, 13-sep-2026, pegando el defecto tal cual salió:
#
#     «ALIMENTOS A MENORES DE EDAD. TIENEN UNA TRIPLE DIMENSIÓN, …»
#     registro 2023835, reconoce que la obligación alimentaria no se reduce a
#     una relación privada entre progenitor y menor, …
#
# El modelo escribe «…, de rubro «RUBRO», registro 2023835, reconoce que…» y el
# compositor corta por el rubro. La cola arranca con el RESTO DE LA FICHA y a
# continuación viene el verbo de la oración decapitada.
#
# El arreglo de abajo existía y no la cogía: `_RX_ARRANQUE_NOMINAL` incluye la
# palabra «registro», así que la tomaba por residuo de ficha —«registro digital
# 179849.»— y la dejaba intacta. La diferencia entre las dos no está en la
# primera palabra: está en si DETRÁS de la ficha sigue una oración.
#
# Se poda la ficha y se decide sobre lo que queda. Podarla no pierde nada: el
# registro ya se dijo en el párrafo de entrada que escribe `escribir_cita`
# —«de registro digital 2023835, de rubro…»— y repetirlo dos renglones después
# era, además de la causa del corte, una redundancia.
#
# EL FRENO: exige coma o punto y coma al final. «registro digital 179849.»
# termina en punto, no se poda, y sigue cayéndose por el filtro de longitud del
# llamador, que es lo que ya hacía bien.
_RX_FICHA_AL_FRENTE = re.compile(
    r"^(?:registro(?:\s+digital)?|p[áa]gina|tomo|libro|volumen|[ée]poca|"
    r"tesis|n[úu]mero|clave|instancia|materia|localizaci[óo]n|"
    r"gaceta(?:\s+del\s+semanario\s+judicial(?:\s+de\s+la\s+federaci[óo]n)?)?|"
    r"semanario(?:\s+judicial(?:\s+de\s+la\s+federaci[óo]n)?)?)"
    # EL PUNTO DE LA ABREVIATURA SÍ, EL DEL FINAL DE FRASE NO. La ficha real
    # lleva puntos dentro —«tesis 1a./J. 44/2021»— así que excluirlos dejaba
    # fuera la forma más común de citar en México. Pero tragarse un punto de
    # cierre haría que la poda saltara a la oración siguiente y se comiera texto
    # bueno. Se distingue por lo que viene detrás: un punto seguido de mayúscula
    # —o del final— cierra frase; «J. 44» no.
    r"(?:[^,;.]|\.(?!\s*(?:[A-ZÁÉÍÓÚÑ]|$))){0,80}[,;]\s*", re.I)


# EL ÓRGANO PEGADO DETRÁS DEL RUBRO TAMBIÉN ES FICHA. ADC 93/2026 v10: el
# modelo escribió «…de rubro «DEMANDA DE NULIDAD…», de la Segunda Sala de la
# Suprema Corte de Justicia de la Nación.»; la cita se rehízo desde el acervo
# —que ya nombra la Sala— y la coletilla quedó sola como párrafo, dos veces en
# el mismo estudio. Once palabras pasan el filtro de «más de seis es oración».
_RX_ORGANO_AL_FRENTE = re.compile(
    r"^(?:de\s+la|del|de)\s+(?:"
    r"(?:Primera|Segunda)\s+Sala(?:\s+de\s+la\s+Suprema\s+Corte\s+de\s+Justicia"
    r"(?:\s+de\s+la\s+Naci[óo]n)?)?"
    r"|(?:Tribunal\s+)?Pleno(?:\s+de\s+la\s+Suprema\s+Corte\s+de\s+Justicia"
    r"(?:\s+de\s+la\s+Naci[óo]n)?)?"
    r"|(?:un|el|los)\s+Tribunal(?:es)?\s+Colegiados?(?:\s+de\s+Circuito)?"
    r"(?:\s+en\s+Materias?\s+[^,;.]{0,80})?"
    r")\s*(?:[,;.]\s*|$)", re.I)


def _sin_ficha_al_frente(c: str) -> str:
    """Poda los trozos de ficha pegados al principio de la cola."""
    for _ in range(4):                      # «registro X, página Y, tomo Z,»
        n = _RX_FICHA_AL_FRENTE.sub("", c, count=1).lstrip(" ,;:")
        n = _RX_ORGANO_AL_FRENTE.sub("", n, count=1).lstrip(" ,;:.")
        if n == c:
            break
        c = n
    return c


def _con_sujeto_tras_cita(cola: str, tesis: dict) -> str:
    """Le devuelve el sujeto a la media frase que quedó debajo de la cita."""
    c = (cola or "").lstrip()
    podada = _sin_ficha_al_frente(c)
    if podada != c:
        # SI AL PODAR NO QUEDA ORACIÓN, era ficha y nada más: se va entera. Con
        # tres palabras o menos no hay sujeto que reponer ni frase que salvar.
        if len(podada.split()) <= 3:
            return ""
        c = podada
    if not c or not c[0].islower() or _RX_ARRANQUE_ATADO.match(c):
        return c
    # NO A LOS FRAGMENTOS. Lo que sigue a la cita no siempre es media oración:
    # a veces es un resto de la ficha —«registro digital 179849.»—, y ponerle
    # sujeto produce «La jurisprudencia en cita registro digital 179849.», que
    # es peor que el hueco. Salió en la primera corrida con este arreglo.
    #
    # Dos filtros: una oración de verdad tiene más de seis palabras, y empieza
    # por VERBO, no por un sustantivo de la ficha ni por una preposición.
    if len(c.split()) <= 6 or _RX_ARRANQUE_NOMINAL.match(c):
        return c
    nombre = ("La jurisprudencia en cita"
              if "JURISPRUDENCIA" in str(tesis.get("tipo") or "").upper()
              else "La tesis en cita")
    return f"{nombre} {c}"


# CUÁNTO SE PARECE UNA FRASE A LA TESIS PARA SER ECO. Con el texto en el
# cuerpo, el 72 % de sus palabras: el lector acaba de leerla arriba. Con el
# texto AL PIE —lo normal: toda tesis de más de `MAX_PALABRAS_TESIS_CUERPO`
# palabras baja a la nota—, el lector del cuerpo sólo ve el rubro, y la frase
# que dice qué sostiene la tesis es justo lo que la hace hablar (AR 631/2025,
# 28-sep-2026: «después de citar tesis hay que hacerlas hablar»). Ahí sólo se
# borra la copia casi literal, que duplica la nota sin decir nada.
UMBRAL_ECO_EN_CUERPO = 0.72
UMBRAL_ECO_AL_PIE = 0.92


def _texto_al_pie(t: dict) -> bool:
    """¿El texto de esta tesis baja a la nota? La misma condición que usa
    `escribir_cita`: si cambia una, cambia la otra."""
    return len(_sin_coletilla_de_organo((t or {}).get("texto") or "").split()) \
        > MAX_PALABRAS_TESIS_CUERPO


def _sin_eco(texto: str, cuerpo_tesis: str, al_pie: bool = False) -> str:
    """Quita del párrafo las frases que repiten la tesis ya transcrita.

    Pedirlo en el prompt no basta: se dijo en el cuerpo y al final, y aun así
    el modelo vuelve a contar la tesis en una de cada dos corridas. Se hace
    aquí, que es donde no falla, y QUIRÚRGICAMENTE: se borran las frases que
    repiten y se conserva lo que aplica el criterio al caso, que es lo único
    que el lector no tiene ya delante.

    `al_pie`: el texto de la tesis bajó a la nota. Entonces la explicación
    propia se queda y sólo se borra la copia casi literal (ver arriba).
    """
    if not (cuerpo_tesis or "").strip():
        return texto
    voc = [set(_norm_palabras(f)) for f in _frases_de(cuerpo_tesis)]
    if not voc:
        return texto
    umbral = UMBRAL_ECO_AL_PIE if al_pie else UMBRAL_ECO_EN_CUERPO
    quedan = []
    for frase in re.split(r"(?<=[.])\s+", texto or ""):
        p = set(_norm_palabras(frase))
        if len(p) >= 8 and any(len(p & v) / max(1, len(p)) > umbral for v in voc):
            continue                       # el lector acaba de leerla arriba
        quedan.append(frase)
    return " ".join(x for x in quedan if x.strip()).strip()


# EL MISMO ECO, PERO CON LOS PRECEPTOS. El dictamen del 382/2024 v6 lo contó
# cinco veces seguidas: el modelo escribe un extracto entrecomillado del
# artículo 840 —«fracciones IV, VI y VII»— y el compositor pega debajo el
# artículo íntegro; luego el 47, el 815, el 784 y el 48, todos dos veces. El
# lector se encuentra lo mismo dos renglones después y el proyecto engorda de
# paja legislativa.
#
# No se arregla en el prompt: al modelo hay que dejarle citar el trozo que le
# interesa, porque es su razonamiento. Se arregla al componer, y en este orden:
# si el documento va a transcribir el artículo entero justo debajo, el extracto
# entrecomillado de arriba SOBRA y se borra.
_RX_ENTRECOMILLADO = re.compile(r"[«\"“]([^»\"”]{40,1800})[»\"”]")


# La fórmula de remate, en las formas en que el modelo la escribe. Se exige el
# verbo de desenlace además del giro inicial: «en consecuencia» abre muchas
# frases con contenido y no se puede borrar por su primera palabra.
# La fórmula de remate. Se exige el giro de cierre Y un verbo de desenlace:
# «en consecuencia» abre muchas frases con contenido y no se puede borrar por su
# primera palabra.
_RX_REMATE = re.compile(
    r"^\s*(?:en\s+ese\s+sentido|en\s+consecuencia|por\s+(?:lo\s+)?"
    r"(?:tanto|ello|consiguiente)|as[íi]\s+las\s+cosas)\s*,?[^.]{0,200}?"
    r"\b(?:lo\s+procedente\s+es|procede\s+(?:confirmar|negar|sobreseer|declarar)|"
    r"debe(?:n)?\s+(?:confirmarse|negarse|sobreseerse|declararse)|"
    r"se\s+(?:confirma|niega|sobresee|declara))\b[^.]{0,200}\.\s*$",
    re.I)

# Más allá de esto no es un remate: es un párrafo con razonamiento dentro que
# empieza con el mismo giro. Medido sobre el cierre real del 410/2026, que son
# 16 palabras.
MAX_PALABRAS_REMATE = 45


# ═══════════════════════════════════════════════════════════════════════════
# UNA FRASE QUE REMITE A UNA TRANSCRIPCIÓN QUE NO ESTÁ
# ═══════════════════════════════════════════════════════════════════════════
# David, sobre la revisión fiscal 91/2025: «El artículo 38 del Código Fiscal de
# la Federación establece lo siguiente.» —y debajo, en vez del texto, otro
# párrafo—. Y «Del precepto transcrito deriva que…» sobre un precepto que sólo
# está al pie. «Esos errores no permitirán tener un proyecto firmable».
#
# LA CAUSA ESTÁ ARRIBA, no aquí: la arquitectura del prompt le ordenaba
# transcribir el precepto entre comillas mientras el documento lo bajaba a la
# nota. El modelo obedecía. Eso ya se corrigió en `fase6_estudio`.
#
# Esto es la red por debajo, y son DOS reparaciones de distinta naturaleza:
#
# (1) QUITAR «TRANSCRITO» es puramente sustractivo y no puede estropear nada:
#     «Del precepto transcrito deriva que…» → «Del precepto deriva que…». La
#     frase queda igual de correcta y deja de mandar al lector a buscar algo
#     que no existe. Medido: aparece en LOS 39 proyectos generados.
#
# (2) FUNDIR EL ANUNCIO CON SU DERIVACIÓN sí reescribe, y por eso va con freno.
#     Sólo cuando el párrafo TERMINA en la fórmula que anuncia y el siguiente
#     EMPIEZA derivando de ella: entonces son una sola frase partida en dos, y
#     se juntan. Si no encajan en el molde NO SE TOCA NADA: un anuncio suelto
#     se ve y se corrige; una frase inventada se firma sin mirarla.
_RX_TRANSCRITO_ADJ = re.compile(
    r"\s+(?:antes\s+|ya\s+|anteriormente\s+)?transcrit[oa]s?\b", re.I)

# «…establece lo siguiente.» al FINAL del párrafo, con su sujeto delante.
_RX_ANUNCIO_VACIO = re.compile(
    r"(?P<sujeto>[^.;]{6,220}?)\s+"
    r"(?P<verbo>establece|dispone|se[ñn]ala|prev[ée]|indica|refiere|dice|"
    r"reza|expresa|contempla)\s+lo\s+siguiente\s*[.:]\s*$", re.I)

# «Del precepto transcrito deriva que…», «De esa disposición se desprende que…»
_RX_DERIVACION = re.compile(
    r"^(?:del?\s+(?:l[ao]s?\s+|es[ae]\s+|dich[ao]s?\s+|ah[íi]\s+)?"
    r"(?:precepto|disposici[óo]n|numeral|art[íi]culo|norma|fracci[óo]n|"
    r"texto|transcripci[óo]n|anterior|lo\s+anterior)[^.]{0,90}?|"
    r"conforme\s+a\s+(?:ese|dicho|dicha|esa)[^.]{0,60}?)\s+"
    r"(?:deriva|se\s+desprende|se\s+advierte|se\s+sigue|se\s+obtiene|"
    r"resulta|se\s+colige|se\s+extrae)\s+(?:la\s+regla\s+de\s+)?que\s+",
    re.I)


def _sin_transcrito(t: str) -> str:
    """Quita el adjetivo que promete una transcripción que está en la nota."""
    return _RX_TRANSCRITO_ADJ.sub("", t or "")


def _sin_anuncio_vacio(parrafos):
    """Funde «X establece lo siguiente.» con «Del precepto deriva que Y».

    Devuelve la lista con los dos párrafos convertidos en uno. Si el par no
    encaja en el molde, la lista vuelve intacta.
    """
    if not isinstance(parrafos, (list, tuple)):
        return parrafos
    fuera, i = [], 0
    ps = [str(x) for x in parrafos]
    while i < len(ps):
        a = ps[i].strip()
        b = ps[i + 1].strip() if i + 1 < len(ps) else ""
        ma = _RX_ANUNCIO_VACIO.search(a)
        mb = _RX_DERIVACION.match(_sin_transcrito(b)) if b else None
        if ma and mb:
            resto = _sin_transcrito(b)[mb.end():].strip()
            if resto:
                cabeza = a[:ma.start()].rstrip()
                sujeto = ma.group("sujeto").strip()
                verbo = ma.group("verbo").lower()
                # NO SE FUNDE SI EL VERBO SE REPITE. «dispone que … dispone
                # lo siguiente» fue mi primer arreglo de este defecto, y era
                # peor que el defecto. Si al unir las dos frases el verbo
                # vuelve a aparecer al principio de la derivación, se dejan
                # separadas y el anuncio se ve —que es lo que se corrige a
                # mano en diez segundos—.
                if re.match(rf"\s*(?:se\s+)?{verbo}\b", resto, re.I):
                    fuera.append(_sin_transcrito(ps[i]))
                    i += 1
                    continue
                unido = f"{sujeto} {verbo} que {resto[0].lower()}{resto[1:]}"
                fuera.append((cabeza + " " + unido).strip() if cabeza else unido)
                i += 2
                continue
        fuera.append(_sin_transcrito(ps[i]))
        i += 1
    return fuera


def _sin_remate_duplicado(texto: str) -> str:
    """Quita la frase de cierre del modelo, no su recapitulación.

    Debajo del estudio el documento añade la fórmula que corresponde al tipo
    —«En ese sentido, ante la ineficacia de los agravios planteados, lo
    procedente es confirmar la sentencia recurrida»— y el modelo escribe la
    suya por iniciativa propia. En la revisión 410/2026 el proyecto acabó con
    las dos, seguidas, diciendo lo mismo.

    SE QUITA LA FRASE, NO EL PÁRRAFO. La recapitulación del modelo dice qué
    agravio es infundado y por qué, y eso es sustancia. Sólo se recorta el
    remate: tiene que empezar con el giro de cierre, llevar un verbo de
    desenlace, y no pasar de MAX_PALABRAS_REMATE palabras. Un párrafo más largo que eso
    razona, aunque empiece igual.
    """
    # EL CUERPO DEL ESTUDIO ES UNA LISTA DE PÁRRAFOS, no una cadena. Se acepta
    # cualquiera de las dos: la lista es lo que llega desde `componer`, y la
    # cadena es cómoda para probar.
    if isinstance(texto, (list, tuple)):
        parrafos = [str(x) for x in texto]
        if not parrafos:
            return texto
        recortado = _sin_remate_duplicado(parrafos[-1].rstrip())
        if recortado == parrafos[-1].rstrip():
            return texto
        fuera = parrafos[:-1]
        if recortado.strip():          # quedaba recapitulación: se conserva
            fuera.append(recortado)
        return type(texto)(fuera) if isinstance(texto, tuple) else fuera

    t = (texto or "").rstrip()
    if not t:
        return texto
    # El último párrafo, y dentro de él la última frase: el remate puede venir
    # solo o pegado al final de la recapitulación.
    corte = t.rfind("\n")
    cabeza, ultimo = (t[:corte + 1], t[corte + 1:]) if corte >= 0 else ("", t)
    frases = re.split(r"(?<=[.])\s+", ultimo.strip())
    if not frases:
        return texto
    if len(frases[-1].split()) <= MAX_PALABRAS_REMATE and _RX_REMATE.match(frases[-1]):
        quedan = " ".join(frases[:-1]).strip()
        # Si el remate era TODO el último párrafo, se va el párrafo entero.
        return (cabeza + quedan).rstrip() if quedan else cabeza.rstrip()
    return texto


# EL REMIENDO DEL ARTÍCULO HUÉRFANO SE RETIRÓ, y conviene decir por qué.
#
# El huérfano —«El artículo 14 de la Constitución Política … .» sin verbo— lo
# producía `_sin_extracto_repetido` al podar el introductor. Ahí se arregló:
# el verbo se conserva y la nota al pie hace de complemento.
#
# El parche que había aquí actuaba DESPUÉS y hacía daño: en el v9 convirtió
# «el artículo 76 … dispone que el órgano jurisdiccional…» en «…dispone que el
# órgano jurisdiccional DISPONE LO SIGUIENTE», y su segunda versión habría
# fundido frases que ya estaban bien, borrando la introducción de la nota.
#
# Un remiendo que puede estropear texto correcto es peor que el defecto que
# arregla: el defecto se ve y se corrige a mano; el destrozo se firma. Van dos
# veces en este mismo punto.


def _sin_extracto_repetido(texto: str, preceptos: list) -> str:
    """Quita del párrafo los entrecomillados del artículo que se va a transcribir."""
    if not preceptos or not (texto or "").strip():
        return texto
    vocablos = []
    for _num, n in preceptos:
        pal = set(_norm_palabras(str(n.get("texto") or "")))
        if len(pal) >= 12:
            vocablos.append(pal)
    if not vocablos:
        return texto
    fuera = texto
    for m in list(_RX_ENTRECOMILLADO.finditer(texto)):
        p = set(_norm_palabras(m.group(1)))
        if len(p) < 8:
            continue
        # El extracto está CONTENIDO en el artículo: casi todas sus palabras
        # aparecen en el texto que se va a transcribir. Ese es el eco.
        if any(len(p & v) / max(1, len(p)) > 0.80 for v in vocablos):
            fuera = fuera.replace(m.group(0), "")
    if fuera == texto:
        return texto
    # EL VERBO SE QUEDA. AHORA HAY NOTA AL PIE.
    #
    # Esto podaba el introductor entero —«…, que dispone: «texto».» quedaba en
    # «El artículo 14 de la Constitución Política de los Estados Unidos
    # Mexicanos.»— y tenía razón mientras debajo venía el bloque transcrito: la
    # frase hacía de rótulo del bloque y se leía bien.
    #
    # Al bajar el precepto a la NOTA AL PIE eso dejó de valer, y es la causa de
    # los «artículos huérfanos» que David ha señalado TRES veces. Con la nota,
    # el verbo no sobra: es lo que la anuncia. «El artículo 14 … dispone lo
    # siguiente.¹» se lee exactamente como se cita en una sentencia.
    #
    # Así que el introductor no se poda: se NORMALIZA. Lo que fuera —«, que
    # dispone:», «el cual establece:», «que señala lo siguiente:»— se convierte
    # en un verbo con su punto, y la nota al pie hace de complemento.
    fuera = re.sub(
        r"[,;]?\s*(?:en\s+la\s+parte\s+conducente[,\s]*)?"
        r"(?:el\s+cual|la\s+cual|que|y\s+que|donde)?\s*"
        r"(?P<v>dispone|establece|se[ñn]ala|prev[ée]|dice|reza|indica|prescribe)"
        r"(?:\s+lo\s+siguiente)?\s*:?\s*(?=[.;]|$)",
        lambda m: f" {m.group('v').lower()} lo siguiente", fuera, flags=re.I)
    fuera = re.sub(r"\s*[,:;]\s*(?=[.;])", "", fuera)
    fuera = re.sub(r"\s{2,}", " ", fuera).strip()
    fuera = re.sub(r"\s+\.", ".", fuera)
    return fuera.strip(" ,;:")


def _frases_de(t: str) -> list:
    return [f for f in re.split(r"(?<=[.])\s+", t or "") if len(f.split()) >= 8]


def _norm_palabras(t: str) -> list:
    import unicodedata
    t = unicodedata.normalize("NFKD", (t or "").lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9 ]+", " ", t).split()


# ═══ LOS EFECTOS LOS ESCRIBE EL MODELO, NO LA PLANTILLA ════════════════════
# Auditando el proyecto salió el defecto más caro de todos: el modelo había
# redactado SIETE efectos concretos —«el quinto efecto consiste en que obtenga
# información oficial sobre el importe, periodicidad y condiciones de pago de
# la pensión por orfandad»— y el compositor los tiraba para poner en su lugar
# «dicte otra en la que atienda los lineamientos de esta ejecutoria».
#
# Es justo el efecto que NO se puede ejecutar sin interpretarlo, que es lo que
# el corpus prohíbe. La responsable recibe la ejecutoria y no sabe qué hacer.
# Se usan los del modelo; la fórmula de plantilla queda de respaldo para cuando
# no los haya escrito.
# MÁS APERTURAS, Y EL RÓTULO FIJO. La v5 del ADC 93/2026 escribió «La
# concesión del amparo exige que la Sala responsable: a) deje insubsistente…»
# y este patrón sólo conocía «debe producir los efectos»: los efectos se
# quedaron dentro del estudio y el considerando de Efectos salió con la
# fórmula genérica. Desde hoy el prompt pide el rótulo «EFECTOS DE LA
# CONCESIÓN» en su propia línea, y aquí se reconoce junto con las formas en
# que el modelo los venía abriendo.
_RX_ROTULO_EFECTOS = re.compile(
    r"^\s*(?:\*\*|__)?\s*efectos(?:\s+de\s+la\s+(?:concesi[óo]n|protecci[óo]n\s+constitucional))?"
    r"\s*(?:\*\*|__)?\s*[.:]?\s*$", re.I)
_RX_INICIO_EFECTOS = re.compile(
    r"^\s*(?:\*\*|__)?\s*efectos(?:\s+de\s+la\s+(?:concesi[óo]n|protecci[óo]n\s+constitucional))?"
    r"\s*(?:\*\*|__)?\s*[.:]?\s*$|"
    r"(?:^|\s)(?:por\s+tanto,?\s+|en\s+consecuencia,?\s+)?la\s+concesi[óo]n\s+del\s+amparo\s+"
    r"(?:debe\s+producir\s+los\s+efectos|exige\s+que|implica\s+que|obliga\s+a|"
    r"tendr[áa]\s+(?:por|como)\s+efecto|se\s+traduce\s+en)|"
    r"^\s*los\s+efectos\s+de\s+la\s+(?:concesi[óo]n|protecci[óo]n)|"
    r"^\s*el\s+primer\s+efecto\s+consiste|"
    r"^\s*(?:en\s+consecuencia,?\s+|por\s+tanto,?\s+)?(?:el\s+amparo\s+se\s+concede|"
    r"procede\s+conceder\s+el\s+amparo|la\s+protecci[óo]n\s+constitucional\s+(?:deber[áa]|se\s+concede))"
    r"[^.]{0,80}para\s+(?:el\s+)?efecto\s+de\s+que", re.I)
_RX_UN_EFECTO = re.compile(
    r"^\s*(?:el\s+)?(?:primer|segundo|tercer|cuarto|quinto|sexto|s[ée]ptimo|"
    r"octavo)\s+efecto\b", re.I)


def partir_efectos(estudio: list) -> tuple:
    """(estudio sin los efectos, párrafos de efectos). Vacío si no los escribió."""
    if not estudio:
        return list(estudio or []), []
    corte = None
    for i, t in enumerate(estudio):
        if _RX_INICIO_EFECTOS.search(t or "") or _RX_UN_EFECTO.match(t or ""):
            corte = i
            break
    if corte is None:
        return list(estudio), []
    cuerpo = list(estudio[:corte])
    efectos = [x for x in estudio[corte:] if (x or "").strip()]
    # EL RÓTULO NO ES UN EFECTO: el considerando ya lleva el suyo («Efectos.»).
    if efectos and _RX_ROTULO_EFECTOS.match(efectos[0]):
        efectos = efectos[1:]
    # El párrafo final de cierre —«por lo que procede conceder el amparo…»— no
    # es un efecto: cierra el estudio y ya lo pone el resolutivo.
    while efectos and re.search(r"procede\s+conceder\s+el\s+amparo",
                                efectos[-1], re.I) and len(efectos) > 1:
        efectos.pop()
    return cuerpo, efectos


# ═══ LAS ÓRDENES DE EFECTOS CUELGAN DE «DEBERÁ:» ══════════════════════════
# David, 27-sep-2026: el considerando salía «Efectos. 1. Deje insubsistente…»
# —la primera orden pegada al rótulo, sin fundamento y sin sujeto—, y debe
# decir «Efectos. Con fundamento en el artículo 77 de la Ley de Amparo, la
# autoridad responsable deberá:» seguido de tres a cinco órdenes. La apertura
# la pone el documento (`tipos_asunto.APERTURA_EFECTOS`); el modelo escribe las
# órdenes, y aquí se ordenan: una por párrafo, numeradas, en infinitivo.
#
# EL INFINITIVO SE PIDE EN EL PROMPT Y SE ASEGURA AQUÍ, porque una sesión
# anterior —o la v1— trae las órdenes en subjuntivo, y «deberá: 1. Deje» no
# concuerda. La conversión es sólo del PRIMER verbo de cada orden de primer
# nivel, y sólo de los que están en esta tabla: un verbo que no conozco se deja
# como vino (un subjuntivo que sobrevive se lee; un infinitivo inventado, no).
# Los incisos de segundo nivel NO se tocan: cuelgan de «en la que:» y ahí el
# subjuntivo es lo correcto («emitir una nueva en la que: a) reitere…»).
_INFINITIVO = {
    "deje": "dejar", "dicte": "dictar", "emita": "emitir", "reitere": "reiterar",
    "analice": "analizar", "examine": "examinar", "estudie": "estudiar",
    "resuelva": "resolver", "reponga": "reponer", "admita": "admitir",
    "corra": "correr", "emplace": "emplazar", "desahogue": "desahogar",
    "abra": "abrir", "valore": "valorar", "funde": "fundar", "motive": "motivar",
    "precise": "precisar", "determine": "determinar", "explique": "explicar",
    "señale": "señalar", "mantenga": "mantener", "conserve": "conservar",
    "confronte": "confrontar", "atienda": "atender", "provea": "proveer",
    "ordene": "ordenar", "realice": "realizar", "practique": "practicar",
    "cite": "citar", "notifique": "notificar", "requiera": "requerir",
    "verifique": "verificar", "continúe": "continuar", "continue": "continuar",
    "subsane": "subsanar", "tome": "tomar", "considere": "considerar",
    "califique": "calificar", "fije": "fijar", "cuantifique": "cuantificar",
    "calcule": "calcular", "cumpla": "cumplir", "restituya": "restituir",
    "devuelva": "devolver", "pague": "pagar", "reintegre": "reintegrar",
    "cancele": "cancelar", "levante": "levantar", "reciba": "recibir",
    "prescinda": "prescindir", "omita": "omitir", "aplique": "aplicar",
    "inaplique": "inaplicar", "declare": "declarar", "reconozca": "reconocer",
    "condene": "condenar", "absuelva": "absolver", "conceda": "conceder",
    "niegue": "negar", "revoque": "revocar", "confirme": "confirmar",
    "modifique": "modificar", "decrete": "decretar", "sobresea": "sobreseer",
    "responda": "responder", "conteste": "contestar", "pondere": "ponderar",
    "identifique": "identificar", "individualice": "individualizar",
    "integre": "integrar", "recabe": "recabar", "solicite": "solicitar",
    "remita": "remitir", "deseche": "desechar", "regularice": "regularizar",
    "amplíe": "ampliar", "acuerde": "acordar", "celebre": "celebrar",
    "proceda": "proceder", "siga": "seguir",
    "prosiga": "proseguir", "repare": "reparar", "haga": "hacer",
    "vuelva": "volver", "prevenga": "prevenir", "incluya": "incluir",
    "excluya": "excluir", "tenga": "tener", "reponga": "reponer",
    "dé": "dar", "emplace": "emplazar", "exponga": "exponer",
    "establezca": "establecer", "evalúe": "evaluar", "evalue": "evaluar",
    "abstenga": "abstenerse", "pronuncie": "pronunciarse", "ocupe": "ocuparse",
    "avoque": "avocarse", "allegue": "allegarse",
    "pronúnciese": "pronunciarse", "ocúpese": "ocuparse",
    "absténgase": "abstenerse",
}
# Los que se conjugan con «se» delante: «se pronuncie» → «pronunciarse».
_CON_SE = {"abstenga", "pronuncie", "ocupe", "avoque", "allegue"}

_RX_ITEM_NUM = re.compile(r"^\s*(\d{1,2})\s*[.)\-–]\s*(.+)$", re.S)
_RX_ITEM_LETRA = re.compile(r"^\s*([a-z])\s*\)\s*(.+)$", re.S)
_RX_ITEM_VIÑETA = re.compile(r"^\s*[-•*–]\s+(.+)$", re.S)
# «La autoridad responsable, Segunda Sala…, deberá dejar…» / «Acto seguido,
# deberá emitir…»: el sujeto y el «deberá» ya los pone la apertura.
_RX_DEBERA = re.compile(r"^(?P<pre>[^:;]{0,200}?)\bdeber[áa]\s+(?=\w+(?:ar|er|ir)(?:se)?\b)", re.I)
# La introducción que el propio modelo escribe antes de la lista.
# Dónde empieza la orden dentro de una introducción: el primer verbo conocido
# después de «para el efecto de que», «a fin de que» o «para que», con su sujeto
# en medio («… de que la Sala responsable deje…»).
_RX_ORDEN_EN_INTRO = re.compile(
    r"(?:para\s+(?:el|los)\s+efectos?\s+de\s+que|a\s+fin\s+de\s+que|para\s+que)\s+"
    r"(?:[^,;:]{0,160}?\s)??(?P<se>se\s+)?(?P<v>" + "|".join(
        sorted((re.escape(k) for k in (
            "deje dicte emita reitere analice examine estudie resuelva reponga admita "
            "corra emplace desahogue valore funde precise determine provea ordene "
            "realice practique restituya devuelva pague cancele levante reciba "
            "pronuncie ocupe").split()), key=len, reverse=True)) + r")\b", re.I)
_RX_SOLO_ANUNCIA = re.compile(
    r"^(?:se\s+)?\S+\s+(?:lo\s+siguiente|lo\s+que\s+a\s+continuaci[óo]n\s+se\s+\w+|"
    r"(?:las?|los)\s+(?:siguientes?\s+\w+|\w+\s+siguientes?))\s*[:.]?\s*$", re.I)
_RX_INTRO_EFECTOS = re.compile(
    r":\s*$|para\s+(?:el|los)\s+efectos?\s+(?:siguientes|de\s+que)|lo\s+siguiente\s*[:.]?\s*$",
    re.I)


def _a_infinitivo(texto: str) -> tuple:
    """(texto, convertido). El primer verbo de la orden —o el primero tras una
    frase de enlace corta: «Hecho lo anterior, dicte…» → «…, dictar…»— y los
    que se coordinan con él antes de cualquier subordinada («admitir la
    ampliación y provea» → «y proveer»). Desde el primer «que» no se toca nada:
    «emitir otra en la que examine…» lleva el subjuntivo que le corresponde."""
    t = (texto or "").strip()
    convertido = False
    m = _RX_DEBERA.match(t)
    if m:
        pre = m.group("pre").strip()
        resto = t[m.end():]
        # Si lo que precede es el sujeto («La autoridad responsable, …,»), sobra;
        # si es una frase de enlace corta («Acto seguido,»), se queda. Cualquier
        # otra cosa es un «deberá» DENTRO de la orden («emita otra en la que
        # deberá analizar…») y no se toca.
        sujeto = re.match(r"^(?:la|el)\s+(?:autoridad|sala|juez|jueza|tribunal|junta|"
                          r"responsable|magistrad)", pre, re.I)
        enlace = pre.endswith(",") and len(pre) <= 60
        if not pre or sujeto or enlace:
            t = resto if (not pre or sujeto) else f"{pre} {resto}"
            convertido = True

    def _inf(palabra, se):
        clave = palabra.lower()
        inf = _INFINITIVO.get(clave)
        if not inf or (se and clave not in _CON_SE):
            return None
        return inf

    _PAL = r"[A-Za-zÁÉÍÓÚÑáéíóúñü]+"
    # 1 · EL PRIMER VERBO. Si la orden ya empieza en infinitivo, se respeta.
    m0 = re.match(rf"^(?P<se>se\s+)?(?P<v>{_PAL})\b", t, re.I)
    ya_infinitivo = bool(m0 and not m0.group("se")
                         and re.fullmatch(r"\w+(?:ar|er|ir)(?:se|lo|la|los|las)?", m0.group("v"), re.I))
    if m0 and not ya_infinitivo:
        inf = _inf(m0.group("v"), bool(m0.group("se")))
        if inf:
            t = inf + t[m0.end():]
            convertido = True
        else:
            m1 = re.match(rf"^(?P<pre>[^,;:]{{2,80}},\s*)(?P<se>se\s+)?(?P<v>{_PAL})\b", t, re.I)
            if m1 and not re.search(r"\b(?:que|cual|cuales|donde)\b", m1.group("pre"), re.I):
                inf = _inf(m1.group("v"), bool(m1.group("se")))
                if inf:
                    t = m1.group("pre") + inf + t[m1.end():]
                    convertido = True
                elif not m1.group("se") and re.fullmatch(
                        r"\w+(?:ar|er|ir)(?:se|lo|la|los|las)?", m1.group("v"), re.I):
                    convertido = True       # «Hecho lo anterior, dictar…»
    elif ya_infinitivo:
        convertido = True
    # 2 · LOS COORDINADOS, hasta la primera subordinada.
    corte = re.search(r"\b(?:que|cual|cuales|donde|cuyo|cuya|cuyos|cuyas|cuando)\b", t, re.I)
    cabeza, cola = (t[:corte.start()], t[corte.start():]) if corte else (t, "")

    def _coord(mm):
        inf = _inf(mm.group("v"), bool(mm.group("se")))
        return f"{mm.group('enl')}{inf}" if inf else mm.group(0)
    cabeza = re.sub(rf"(?P<enl>(?:(?:,\s*|\s+)(?:y|e)\s*,\s*[^,;:]{{2,60}},\s*|,\s*(?:y\s+|e\s+)?|\s+(?:y|e)\s+))"
                    rf"(?P<se>se\s+)?(?P<v>{_PAL})\b", _coord, cabeza)
    t = cabeza + cola
    return (t[:1].upper() + t[1:] if t else t), convertido


def _cierra_orden(t: str) -> str:
    """Cada orden termina en punto; la que abre incisos, en dos puntos."""
    t = (t or "").rstrip()
    if t.endswith(":"):
        return t
    t = re.sub(r"[;,]\s*(?:y|e)?\s*$", "", t).rstrip()
    return t if t.endswith((".", "»", "”", "\"")) else t + "."


def componer_efectos(parrafos: list) -> tuple:
    """(órdenes, avisos). Las órdenes de primer nivel, numeradas «1.», «2.»… y
    en infinitivo; los incisos y los párrafos de continuación, en su sitio.
    Vacío si no hay una sola orden reconocible (entonces los efectos vinieron
    en prosa y se escriben como vinieron)."""
    ps = [str(x).strip() for x in (parrafos or []) if str(x or "").strip()]
    if not ps:
        return [], []
    hay_num = any(_RX_ITEM_NUM.match(x) for x in ps)
    hay_letra = any(_RX_ITEM_LETRA.match(x) for x in ps)
    hay_viñeta = any(_RX_ITEM_VIÑETA.match(x) for x in ps)

    def _primer_nivel(x):
        if hay_num:
            m = _RX_ITEM_NUM.match(x)
        elif hay_letra:
            m = _RX_ITEM_LETRA.match(x)
        elif hay_viñeta:
            m = _RX_ITEM_VIÑETA.match(x)
            return m.group(1) if m else None
        else:
            return None
        return m.group(2) if m else None

    # LA INTRODUCCIÓN DEL MODELO. Si sólo introduce —«para los efectos
    # siguientes:», «realice lo siguiente:»— sobra: la apertura del documento
    # dice ya quién debe y con qué fundamento. PERO SI TRAE LA ORDEN —«…para el
    # efecto de que la Sala deje insubsistente la sentencia y dicte otra en la
    # que valore la pericial»—, borrarla era perder la orden principal sin aviso
    # (revisión del código, 27-sep-2026): se vuelve la primera orden, desde su
    # verbo. Y si esa orden abre incisos, los incisos cuelgan de ella.
    intro = False
    orden_intro = ""
    while ps and _primer_nivel(ps[0]) is None and len(ps[0]) < 400 \
            and _RX_INTRO_EFECTOS.search(ps[0]):
        m_or = _RX_ORDEN_EN_INTRO.search(ps[0])
        if m_or and not orden_intro:
            _o = ps[0][m_or.start("v") - (len(m_or.group("se") or "")):].strip()
            # «realice lo siguiente:» no es una orden: sólo anuncia la lista.
            if not _RX_SOLO_ANUNCIA.match(_o):
                orden_intro = _o
        ps, intro = ps[1:], True
    if orden_intro:
        if not hay_num and orden_intro.rstrip().endswith(":"):
            # Los incisos cuelgan de la orden que los abre: el primer nivel son
            # la orden de la introducción y los párrafos sin marca.
            hay_num = True
            ps = [f"1. {orden_intro}"] + [
                x if _RX_ITEM_LETRA.match(x) else
                (x if _RX_ITEM_NUM.match(x) else f"{i}. {x}")
                for i, x in enumerate(ps, 2)]
        elif hay_num:
            ps = [f"0. {orden_intro}"] + ps
        elif hay_letra or hay_viñeta:
            ps = [("a) " if hay_letra else "- ") + orden_intro] + ps
        else:
            ps = [orden_intro] + ps
    if not (hay_num or hay_letra or hay_viñeta):
        # SIN MARCAS, UNA ORDEN POR PÁRRAFO —así las escribe el ADA 767/2025:
        # «realice lo siguiente:» y tres párrafos sin número—. Sólo si hubo esa
        # introducción o son varias; un párrafo suelto es prosa.
        if not ps or not (intro or len(ps) > 1):
            return [], []
        _primer_nivel = (lambda x: x)
    ordenes, n, sin_convertir = [], 0, 0
    for x in ps:
        cuerpo = _primer_nivel(x)
        if cuerpo is None:
            # Inciso o continuación de la orden anterior: se queda como vino.
            ordenes.append(x)
            continue
        n += 1
        cuerpo, ok = _a_infinitivo(cuerpo)
        if not ok and not re.match(r"^\S+(?:ar|er|ir)(?:se|lo|la|los|las)?\b", cuerpo, re.I):
            sin_convertir += 1
        ordenes.append(f"{n}. {_cierra_orden(cuerpo[:1].upper() + cuerpo[1:])}")
    avisos = []
    import tipos_asunto as _ta_ef
    if n > _ta_ef.EFECTOS_MAX:
        avisos.append(
            f"LOS EFECTOS SON {n} ÓRDENES y lo habitual es de "
            f"{_ta_ef.EFECTOS_MIN} a {_ta_ef.EFECTOS_MAX}. Fusione las que recaen "
            f"sobre el mismo acto o la misma decisión: juntas deben comprender el "
            f"objetivo de la concesión, no repetir el estudio.")
    palabras = sum(len(o.split()) for o in ordenes)
    larga = max((len(o.split()) for o in ordenes), default=0)
    if palabras > 250 or larga > 60:
        avisos.append(
            f"LOS EFECTOS SUMAN {palabras} PALABRAS"
            + (f" y alguna orden pasa de sesenta" if larga > 60 else "")
            + ". Cada orden dice qué hacer, no por qué: las razones ya están en "
              "el estudio. Acórtelos antes de firmar.")
    if sin_convertir:
        avisos.append(
            "REVISE LA CONCORDANCIA DE LOS EFECTOS: cuelgan de «la autoridad "
            "responsable deberá:» y alguna orden no empieza en infinitivo.")
    return ordenes, avisos


# ══════════════════════════════════════════════════════════════════════════
# EL ESTUDIO DE LOS CONCEPTOS DE VIOLACIÓN ES UN CONSIDERANDO, NO UN SUBTÍTULO
# ══════════════════════════════════════════════════════════════════════════
# David: «implementa el considerando con su ordinal para los conceptos de
# violación».
#
# Cuando la revisión levanta el sobreseimiento, el colegiado asume jurisdicción
# (artículo 93, fracción I) y resuelve lo que el juzgado no resolvió. El prompt
# ya le pedía al modelo abrir «un apartado nuevo», y el modelo lo abría —pero
# como subtítulo en negrita DENTRO del Estudio, porque ahí es donde el
# compositor mete todo lo que devuelve. Un subtítulo no es un considerando: no
# lleva ordinal, y en un engrose el ordinal es lo que dice que ahí empieza otra
# cosa que se resolvió.
#
# NO SE LE PIDE AL MODELO QUE NUMERE. Los ordinales se calculan al final sobre
# la lista de apartados —esa regla ya estaba y es la buena—; lo único que
# faltaba era que este estudio ENTRARA en la lista en vez de quedarse dentro
# del apartado anterior. Se corta por donde el propio modelo puso el rótulo.
_RX_SUB_CONCEPTOS = re.compile(
    r"^\s*(?:\*\*|__)?\s*(?:[IVXLC]+\.|\d{1,2}[.)])?\s*"
    r"(?:estudio\s+de\s+los\s+|an[áa]lisis\s+de\s+los\s+|los\s+)?"
    r"conceptos\s+de\s+violaci[óo]n\s*(?:\*\*|__)?\s*[.:]?\s*$", re.I)


def partir_conceptos(cuerpo: list) -> tuple:
    """(estudio de los agravios, estudio de los conceptos de violación).

    El segundo va vacío salvo que el modelo haya abierto el apartado, que es
    sólo cuando el recurso levanta el sobreseimiento. Si detrás del rótulo no
    hay cuerpo de verdad —dos párrafos al menos— no se parte: más vale un
    subtítulo suelto que un considerando hueco con su ordinal gastado.
    """
    for i, t in enumerate(cuerpo or []):
        if _RX_SUB_CONCEPTOS.match((t or "").strip()):
            resto = [x for x in cuerpo[i + 1:] if (x or "").strip()]
            if len(resto) >= 2:
                return list(cuerpo[:i]), resto
            break
    return list(cuerpo or []), []


# LA PREGUNTA QUE FIJA LA CUESTIÓN NO SE DESCARTA POR CORTA. El compositor
# tira los párrafos de menos de seis palabras —restos de un corte, coletillas
# sueltas— y «1. ¿Era aplicable ese criterio?» son cinco. Habría pedido la
# pregunta expresa y la habría borrado acto seguido.
_RX_ES_PREGUNTA = re.compile(r"^\s*(?:\d{1,2}[.)]\s*)?¿.{10,}\?\s*$", re.S)


def _es_pregunta(t: str) -> bool:
    return bool(_RX_ES_PREGUNTA.match((t or "").strip()))


# LA CITA ENCADENADA. El modelo escribe «Sirve de apoyo la jurisprudencia X, de
# rubro «A», y la jurisprudencia Y, de rubro «B».» Al bajar «A» a su bloque, la
# cola «y la jurisprudencia Y, de rubro «B»» quedaba como párrafo suelto debajo
# de la cita, fuera de toda estructura —David, revisión 322/2025—. Cada tesis
# se anuncia con su propia frase, y la segunda empieza por «También».
_RX_CITA_ENCADENADA = re.compile(
    r"^(?:y|e|as[íi]\s+como|adem[áa]s\s+de)\s+(?=(?:la|el)\s+"
    r"(?:jurisprudencia|tesis|criterio)\b)", re.I)


def _desencadenar(cola: str, tesis: list, contesta: str = "") -> str:
    """Si la cola es «y la jurisprudencia …, de rubro «B»», la devuelve como
    anuncio propio; si no, cadena vacía.

    `contesta`: el arranque de la primera cuando la contestaba (ver
    `es_anuncio_que_contesta`). La segunda de «No resultan aplicables la X… y
    la Y…» no puede salir «También sirve de apoyo» (AR 631/2025): se anuncia
    con «Tampoco resulta aplicable» si la primera se negó, y con el mismo
    arranque si se atribuyó a quien la invocó."""
    c = (cola or "").lstrip(" ,;")
    m = _RX_CITA_ENCADENADA.match(c)
    if not m:
        return ""
    resto = c[m.end():]
    h, _ = tesis_del_rubro(resto, tesis or [])
    if not h:
        # «y la tesis con registro 2020401…» sin rubro: por registro.
        mr = _RX_REGISTRO_EN_PROSA.search(resto)
        if not (mr and any(str(t.get("registro") or "") == mr.group(1) for t in (tesis or []))):
            return ""
    if contesta:
        if re.search(r"\b(?:no|tampoco|ni|sin\s+que)\b|inaplicable", contesta, re.I):
            return "Tampoco resulta aplicable " + resto
        return contesta[:1].upper() + contesta[1:] + " " + resto
    return "También sirve de apoyo " + resto


# LA MEDIA FRASE QUE SIGUE A UNA CITA QUE SE CONTESTA. «No resulta aplicable la
# jurisprudencia…, de rubro «…», porque exige…»: al bajar el rubro a su bloque,
# la cola quedaba «porque exige…», en minúscula y sin sujeto. Se le da el
# arranque que pide su conector; el razonamiento es el del modelo.
_RX_COLA_CAUSAL = re.compile(
    r"^(?:porque|pues|ya\s+que|toda\s+vez\s+que|dado\s+que|puesto\s+que|"
    r"en\s+virtud\s+de\s+que|en\s+tanto\s+que)\b", re.I)
_RX_COLA_ADVERSATIVA = re.compile(r"^(?:pero|mas|empero)\s+", re.I)


def _cola_tras_contestar(cola: str) -> str:
    c = (cola or "").lstrip()
    if not c or not c[0].islower():
        return c
    if _RX_COLA_CAUSAL.match(c):
        return "Ello, " + c
    m = _RX_COLA_ADVERSATIVA.match(c)
    if m:
        return "Sin embargo, " + c[m.end():]
    # «…, de rubro «X», que resolvía…»: el relativo es el criterio.
    if re.match(r"^que\s+(?!se\b)[a-záéíóúñ]", c):
        return "Ese criterio " + c[4:]
    return c


# EL LENGUAJE DEL PROYECTO. El motor le enseña al estudio «la objeción más
# seria» y el modelo la copia con ese nombre. David: «así no se redacta un
# proyecto; se estila "En diverso aspecto, una de las disidencias más
# relevantes…" o "No se pierde de vista la inconformidad de…"». Se pide en el
# prompt y, como todo lo que se pide, se garantiza aquí.
_LENGUAJE_DE_PROYECTO = [
    (re.compile(r"\b[Ll]a objeci[óo]n m[áa]s (?:fuerte|seria|importante|relevante|"
                r"s[óo]lida|grave)(?: (?:a|contra) (?:esta|esa|la) "
                r"(?:soluci[óo]n|conclusi[óo]n|propuesta|determinaci[óo]n))?"
                r"(?: es| consiste en| radica en| estriba en)(?: la de)? que\b"),
     "No se pierde de vista que"),
    (re.compile(r"\b[Ll]a objeci[óo]n m[áa]s (?:fuerte|seria|importante|relevante|"
                r"s[óo]lida|grave)\b"), "la disidencia más relevante"),
    (re.compile(r"\bEsa objeci[óo]n\b"), "Ese planteamiento"),
    (re.compile(r"\besa objeci[óo]n\b"), "ese planteamiento"),
    (re.compile(r"\bEsta objeci[óo]n\b"), "Este planteamiento"),
    (re.compile(r"\besta objeci[óo]n\b"), "este planteamiento"),
    (re.compile(r"\b([Ll]a|[Uu]na|[Dd]icha) objeci[óo]n\b"), r"\1 inconformidad"),
    (re.compile(r"\b([Ll]as) objeciones\b"), r"\1 inconformidades"),
]


def _lenguaje_de_proyecto(texto: str) -> str:
    t = texto or ""
    for rx, rep in _LENGUAJE_DE_PROYECTO:
        t = rx.sub(rep, t)
    return t


def _escribir_estudio(doc, estudio, tesis, notas, normas=None) -> int:
    """Los párrafos del estudio, con sus citas rehechas desde el acervo."""
    # EL ESTUDIO ENTERO, PARA BUSCAR LAS FRACCIONES. El precepto se transcribe
    # donde se le menciona por primera vez, y ahí casi nunca se dice qué
    # fracción interesa: eso se razona páginas después.
    #
    # Medido en la revisión 410/2026: el artículo 79 se transcribió detrás de un
    # párrafo que hablaba de otra cosa —«El asunto se ubica en el segundo
    # supuesto propio de la materia administrativa…»—, así que no había ninguna
    # fracción que leer y salió recortado a 180 palabras, cortado en la IV. Las
    # fracciones VI y VII, que son sobre las que gira el razonamiento, no
    # llegaron nunca al papel. Buscarlas sólo en el párrafo de al lado era
    # mirar por la ventana equivocada.
    _todo_estudio = " ".join(
        str(x) for x in (estudio if isinstance(estudio, (list, tuple)) else [estudio]))
    citadas = 0
    ultima_tesis = None
    ultima_al_pie = False
    _pies_de_ley = len(notas)          # los que ya había antes de este estudio
    transcritos = set()
    transcritas_tesis = set()
    _pendientes = list(estudio or [])
    while _pendientes:
        t = _pendientes.pop(0)
        t = (t or "").strip()
        if not t:
            continue
        # El encabezado ordinal que el modelo se pone a sí mismo sobra: el
        # ordinal lo calcula el compositor y ya está escrito arriba.
        t = re.sub(r"^(?:PRIMERO|SEGUNDO|TERCERO|CUARTO|QUINTO|SEXTO|"
                   r"S[ÉE]PTIMO|OCTAVO|NOVENO)\.\s*(?:Estudio(?:\s+de\s+fondo)?\.\s*)?",
                   "", t)
        t = re.sub(r"^Los\s+(?:conceptos\s+de\s+violaci[óo]n|agravios)\s+son\s+"
                   r"[^.]{3,60}\.\s*", "", t)
        if not t.strip():
            continue
        # NINGÚN PÁRRAFO ARRANCA CON «y la tesis…». Es el resto de una cita
        # encadenada que no se pudo bajar a bloque; se le devuelve el sujeto.
        _mc = _RX_CITA_ENCADENADA.match(t)
        if _mc:
            t = "También sirve de apoyo " + t[_mc.end():]
        elif re.match(r"^(?:y|e)\s+[a-záéíóúñ]", t):
            t = "Asimismo, " + t[2:].lstrip()
        hallada, m_r = tesis_del_rubro(t, tesis or [])
        # LA PROSA QUE VA DELANTE DEL ANUNCIO NO SE PIERDE (AR 631/2025). El
        # prompt pide que la cita cierre su párrafo, y todo lo que había antes
        # de la oración del anuncio se iba con él: `escribir_cita` sólo
        # conserva el verbo de enlace. Se parte en dos y se escriben por orden.
        if hallada and m_r:
            _ini = _inicio_de_oracion(t, m_r.start())
            if _ini > 0 and _arranque_de_cita(t[_ini:m_r.start()]) \
                    and len(t[:_ini].split()) >= 6:
                _pendientes[0:0] = [t[:_ini].strip(), t[_ini:].strip()]
                continue
        # Y UNA TESIS TAMBIÉN. La 169606 se transcribió dos veces en el mismo
        # considerando —una en el marco y otra al contestar el concepto—: el
        # texto íntegro repetido no aporta y alarga la sentencia sin decir nada.
        if hallada and str(hallada.get("registro") or "") in transcritas_tesis:
            # POR REGISTRO Y YA TRANSCRITA (AR 631/2025): el anuncio pelado
            # quedaría colgando con sus dos puntos y nada debajo. Se nombra
            # como ya citada, que es lo que pide el prompt para la segunda vez.
            if m_r is not None and not isinstance(m_r, re.Match) \
                    and not t[m_r.end():].strip(" ,;:.") \
                    and _inicio_de_oracion(t, m_r.start()) == 0:
                _ya = re.sub(r",\s*de rubro(?:\s+y\s+texto)?\s+siguientes?:$", "",
                             anuncio_de(hallada, t[:m_r.start()].rstrip(" ,;:")))
                parrafo_con_citas(doc, _ya + (", ya citado." if re.search(r"\bel criterio\b", _ya)
                                              else ", ya citada."), notas)
                continue
            hallada = None
        if hallada and m_r and citadas < MAX_CITAS_DOCUMENTO:
            transcritas_tesis.add(str(hallada.get("registro") or ""))
            antes = _RX_COLA_ANUNCIO.sub("", t[:m_r.start()].rstrip(" ,;:"))
            cola = t[m_r.end():].lstrip(" ,;:.")
            escribir_cita(doc, hallada, antes.rstrip(" ,;:"), notas)
            citadas += 1
            # ¿La contestaba? Entonces la cita que la sigue en la misma frase
            # y la media frase de detrás se anuncian como lo que son.
            _v_ant = _verbo_de_enlace(antes.rstrip(" ,;:"))
            _contesta = _v_ant if es_anuncio_que_contesta(_v_ant) else ""
            # LA LISTA DE REGISTROS DE UN MISMO ANUNCIO —«Sirven de apoyo los
            # criterios de registros 2026918 y 168958:»—: cada uno, su bloque.
            _lista = [_desencadenar(f"y el criterio de registro {_r}:", tesis, _contesta)
                      for _r in registros_de_la_lista(
                          t, m_r, tesis, str(hallada.get("registro") or ""))]
            _lista = [x for x in _lista if x]
            if _lista:
                _pendientes[0:0] = _lista + ([cola] if len(cola.split()) > 6 else [])
                ultima_tesis = hallada
                ultima_al_pie = _texto_al_pie(hallada)
                continue
            # LA SEGUNDA TESIS DE LA MISMA FRASE se anuncia por su cuenta.
            _otra = _desencadenar(cola, tesis, _contesta)
            if _otra:
                _pendientes.insert(0, _otra)
                ultima_tesis = hallada
                ultima_al_pie = _texto_al_pie(hallada)
                continue
            cola = _sin_eco(cola, hallada.get("texto") or "",
                            al_pie=_texto_al_pie(hallada))
            cola = _con_sujeto_tras_cita(cola, hallada)
            if _contesta or not isinstance(m_r, re.Match):
                cola = _cola_tras_contestar(cola)
            if len(cola.split()) > 6 or _es_pregunta(cola):
                parrafo_con_citas(doc, cola, notas)
            ultima_tesis = hallada
            ultima_al_pie = _texto_al_pie(hallada)
            continue
        if ultima_tesis is not None:
            t = _sin_eco(t, ultima_tesis.get("texto") or "", al_pie=ultima_al_pie)
            ultima_tesis = None
            if len(t.split()) < 6 and not _es_pregunta(t):
                continue
        # Se decide ANTES de escribir qué artículos van a transcribirse, para
        # poder quitar del párrafo el extracto que quedaría repetido debajo.
        _del_parrafo = _preceptos_del_parrafo(t, normas)[:MAX_ARTICULOS_POR_NOTA]
        _preceptos = [(n_, x) for n_, x in _del_parrafo if n_ not in transcritos]
        # EL ECO SOBREVIVÍA CUANDO EL ARTÍCULO YA ESTABA TRANSCRITO. La poda
        # miraba sólo los preceptos que este párrafo va a transcribir DEBAJO;
        # si el compositor ya lo había puesto páginas antes, `_preceptos`
        # quedaba vacío y la copia que el modelo escribió aquí se quedaba.
        #
        # Medido en el ARA 17/2025 generado: el artículo 76 de la Ley de Amparo
        # aparece dos veces, una en el bloque del compositor —entre comillas
        # angulares— y otra en el cuerpo, escrita por el modelo. Es el defecto
        # que el detector de duplicación marcaba y la regla del prompt no
        # bastaba para evitar, porque no es del modelo solo: es de los dos.
        t = _sin_extracto_repetido(t, _del_parrafo)
        if len(t.split()) < 6 and not _es_pregunta(t):
            continue
        p_ = parrafo_con_citas(doc, t, notas)
        # EL PRECEPTO VA AL PIE, COMO LA TESIS.
        #
        # David: «me gusta cómo cita las tesis (con su texto a pie de página),
        # así me gustaría que citara los artículos para una lectura más fluida
        # y sólo referir al contenido del artículo y citarlo a pie de página».
        #
        # Antes se transcribía en bloque, con sangría, detrás del párrafo que
        # lo anuncia. Un proyecto empieza citando el artículo 75, luego el 107
        # constitucional entero, y el lector recorre media página de ley antes
        # de volver al razonamiento. Medido en el 410/2026: 818 palabras —el
        # 16,5% del estudio— eran preceptos transcritos.
        #
        # El texto no se pierde: baja al pie, donde se comprueba si hace falta
        # y no interrumpe si no. Es exactamente el trato que ya tenían las
        # tesis, y por eso se lee bien.
        #
        # `notas_de_articulos` hacía esto y NO LA LLAMABA NADIE: el mecanismo
        # estaba escrito y muerto. Aquí se usa su misma forma de nota, con el
        # recorte a la fracción que el estudio discute, que en el pie sigue
        # importando —el 107 constitucional entero no cabe en una nota—.
        if p_ is not None:
            _pies_p = []
            for num, n_ in _preceptos:
                # UN ARTÍCULO SE TRANSCRIBE UNA VEZ. La clave era (número,
                # ley) y el 48 salió DOS veces porque llegó por dos caminos con
                # el nombre de la ley escrito distinto —«Artículo 48.-» y
                # «Artículo 48.»—. Al lector le da igual de dónde vino: lee lo
                # mismo dos veces seguidas.
                if num in transcritos:
                    continue
                transcritos.add(num)
                # LA FRACCIÓN QUE EL PÁRRAFO CITÓ. Si el texto que anuncia
                # el precepto dice «artículo 107, fracción X», se transcribe
                # ESA fracción y no el artículo entero: es lo que el
                # secretario hace y lo que hace legible el bloque.
                # EN LOS DOS ÓRDENES, Y EN PLURAL. Sólo se reconocía
                # «artículo 79, fracción V». El estudio de la revisión
                # 410/2026 escribió «la fracción V del artículo 79» —que es
                # como se dice más a menudo— y no casó: el artículo se
                # transcribió recortado a 180 palabras y se cortó en la
                # fracción IV, JUSTO UNA ANTES de la que el estudio analizaba.
                # El lector se queda sin ver el precepto en que se apoya el
                # razonamiento que está leyendo.
                # PRIMERO EL PÁRRAFO, LUEGO EL ESTUDIO. Si quien anuncia el
                # precepto ya dice qué fracción le interesa, ésa manda: es la
                # más específica. Si no dice nada —lo habitual—, se busca en
                # todo el estudio, que es donde se razona.
                _fr = _fracciones_citadas(t, num) or \
                      _fracciones_citadas(_todo_estudio, num)
                if len(notas) - _pies_de_ley >= MAX_NOTAS_DE_ARTICULOS:
                    continue          # ocho preceptos al pie ya son bastantes
                _cuerpo = " ".join(str(n_.get("texto") or "").split())
                if not _cuerpo:
                    continue
                # El acervo guarda una migaja delante: «[Ley de Amparo |
                # CAPÍTULO X …] Artículo 79. La autoridad…».
                _ley = n_.get("cuerpo_legal") or n_.get("fuente") or ""
                # POR LA VERJA, igual que las otras dos puertas. Esta era la
                # tercera copia del mismo recorte de cabecera, y por eso el
                # mismo fallo salía por tres sitios.
                _cuerpo = cuerpo_para_transcribir(_cuerpo, num, _ley)
                if not _cuerpo:
                    continue
                _cuerpo = _en_lo_conducente(_cuerpo, _fr)
                _pie = (f"«Artículo {num}. {limpiar_texto_web(_cuerpo)}» — {_ley}".strip()
                        + marca_de_origen(n_))
                if any(_pie in x.split(SEP_NOTA) for x in notas) or _pie in _pies_p:
                    continue
                _pies_p.append(_pie)
            # TODOS LOS DEL PÁRRAFO EN UNA NOTA: una sola llamada, como antes.
            if _pies_p:
                notas.append(SEP_NOTA.join(_pies_p))
                _run_llamada(p_, len(notas))
    return citadas


# ═══ EL ESQUELETO DE CADA TIPO, MEDIDO ═════════════════════════════════════
# amparo directo civil 26 · administrativo 16 · queja 20 (y 153 de recuento) ·
# revisión civil 31 · administrativa 16 · fiscal 28. La regla que vale para
# TODOS: el resumen de lo recurrido NO es considerando; el considerando que
# lleva su nombre es la DISPENSA de transcribirlo.
#
# Lo que cambia de un tipo a otro y rompería una plantilla única:
#   · Los RECURSOS no tienen «Existencia del acto reclamado»: es del amparo.
#     Tampoco la revisión reproduce la de la resolución recurrida (C1,
#     3-oct-2026): lleva su «Procedencia.» antes de la legitimación.
#   · En la QUEJA el cómputo va en PROSA, sin tabla, y la procedencia lleva
#     UNA nota al pie con el artículo 97 de la Ley de Amparo.
#   · El secretario escribe «Trascripción» sin la n: 104 veces contra 12.
#   · El estudio no tiene ordinal fijo: es el último, y su número sale de
#     cuántos apartados le preceden.
ESQUELETO = {
    # ═══════════════════════════════════════════════════════════════════
    # LA TABLA VA EN LOS CUATRO, Y ES UNA DECISIÓN DE PRODUCTO
    # ═══════════════════════════════════════════════════════════════════
    # Yo la había quitado de tres tipos porque el corpus casi no la usa —tabla
    # en 1 de 45 amparos directos, 0 de 21 quejas—. David: «cometí un error al
    # pedirte que coincidieran con mis adelantos. Un plus que tenía el generador
    # era la tabla. Ésa quiero conservarla para todos».
    #
    # Tiene razón y el criterio es distinto del que yo aplicaba: contra el
    # corpus se mide lo que HAY QUE IMITAR —los rótulos, las fórmulas, el orden,
    # el vocabulario— porque ahí el corpus es la autoridad. Pero el corpus no
    # manda sobre lo que el producto puede MEJORAR: si el secretario no dibuja
    # la tabla es porque hacerla a mano cuesta, no porque sobre. La máquina
    # tiene el calendario y la aritmética; regalarle el desglose es justamente
    # lo que ella aporta.
    #
    # La prosa sí se queda corta —«resultó oportuna, a la luz del artículo 17»—
    # porque con la tabla debajo, repetir el cómputo en palabras es decir dos
    # veces lo mismo.
    "amparo_directo": {
        "q": "conceptos de violación",
        "recurrido": "la sentencia reclamada",
        "tabla_computo": True,
        "dispensa": "Acto reclamado y {q}.",
        "legitimacion": "Legitimación y oportunidad.",
        "existencia": True,
        "sub_recurrido": "Sentencia reclamada",
        "adhesivo": "Amparo adhesivo.",
    },
    "amparo_revision": {
        "q": "agravios",
        "recurrido": "la resolución recurrida",
        "tabla_computo": True,
        "dispensa": "Resolución recurrida y {q} de la parte recurrente.",
        "legitimacion": "Legitimación y oportunidad para interponer el recurso.",
        # LA REVISIÓN YA NO REPRODUCE LA EXISTENCIA; LLEVA SU PROCEDENCIA
        # (C1, 3-oct-2026) [siempre]. David: «el considerando de existencia ya
        # no es necesario porque ya viene en la sentencia recurrida; no es usual
        # ni necesario que lo reproduzcamos». La había puesto aquí un adelanto
        # suyo ajustado a mano («el proyecto revisaba una sentencia sin haber
        # dicho antes que consta»); su palabra de hoy la quita. En su lugar, la
        # procedencia del recurso, entre la competencia y la legitimación (los 5
        # engroses del banco que la traen: PRIMERO Competencia, SEGUNDO
        # Procedencia, TERCERO Legitimación y oportunidad), con el inciso del
        # 81 que toca a lo recurrido (`tipos_asunto.procedencia_revision`).
        "existencia": False,
        "procedencia_propia": True,
        "sub_recurrido": "Resolución recurrida",
        "adhesivo": "Revisión adhesiva.",
    },
    "queja": {
        "q": "agravios",
        "recurrido": "el auto recurrido",
        "tabla_computo": True,
        "dispensa": "Trascripción innecesaria del auto recurrido y {q}.",
        "legitimacion": "Legitimación y oportunidad.",
        "existencia": False,
        "procedencia_propia": True,
        "sub_recurrido": "Auto recurrido",
        "adhesivo": "",
    },
    "revision_fiscal": {
        "q": "agravios",
        "recurrido": "la sentencia impugnada",
        "tabla_computo": True,
        "dispensa": "Consideraciones de la sentencia impugnada y {q}.",
        "legitimacion": "Legitimación y oportunidad.",
        "existencia": False,
        "procedencia_propia": True,
        "sub_recurrido": "Sentencia impugnada",
        "adhesivo": "",
    },
}


def esqueleto_de(tipo: str) -> dict:
    return ESQUELETO.get((tipo or "").strip().lower(),
                         ESQUELETO["amparo_directo"])


def _pagina(doc):
    s = doc.sections[0]
    s.page_width, s.page_height = Cm(21.59), Cm(34.03)
    s.left_margin, s.right_margin = Cm(5), Cm(2)
    s.top_margin, s.bottom_margin = Cm(3), Cm(3)
    normal = doc.styles["Normal"]
    normal.font.name = FUENTE
    normal.font.size = TAMANO
    # ═══════════════════════════════════════════════════════════════════════
    # EL DOCUMENTO ESTÁ EN ESPAÑOL, Y HAY QUE DECÍRSELO A WORD
    # ═══════════════════════════════════════════════════════════════════════
    # David: «establece el lenguaje en español del documento y no marque
    # errores al modificar el texto en lenguaje inglés».
    #
    # python-docx crea el .docx a partir de una plantilla en inglés de Estados
    # Unidos, y ese idioma viaja en los `docDefaults`. Word subraya entonces
    # media sentencia en rojo, y el corrector le propone al secretario
    # correcciones inglesas mientras escribe. Se marca es-MX en el estilo
    # Normal y en los valores por omisión, que es de donde hereda todo.
    from docx.oxml.ns import qn as _qn_l

    def _idioma(rpr):
        for _t in ("w:lang",):
            _v = rpr.find(_qn_l(_t))
            if _v is None:
                _v = OxmlElement(_t)
                rpr.append(_v)
            _v.set(_qn_l("w:val"), "es-MX")
            _v.set(_qn_l("w:eastAsia"), "es-MX")
            _v.set(_qn_l("w:bidi"), "es-MX")

    _idioma(normal.element.get_or_add_rPr())
    try:
        _dd = doc.styles.element.find(_qn_l("w:docDefaults"))
        if _dd is not None:
            _rpd = _dd.find(_qn_l("w:rPrDefault"))
            if _rpd is None:
                _rpd = OxmlElement("w:rPrDefault")
                _dd.insert(0, _rpd)
            _rpr = _rpd.find(_qn_l("w:rPr"))
            if _rpr is None:
                _rpr = OxmlElement("w:rPr")
                _rpd.append(_rpr)
            _idioma(_rpr)
    except Exception as _el:
        print(f"   ⚠️ no se pudo fijar el idioma por omisión: {_el}")
    # EL INTERRUPTOR DE PAR/IMPAR VIVE EN settings.xml, y python-docx no lo
    # expone. Sin él, Word ignora el encabezado de página par por mucho que
    # esté escrito en el fichero: se define y no se usa.
    _ajustes = doc.settings.element
    from docx.oxml.ns import qn as _qn
    if _ajustes.find(_qn("w:evenAndOddHeaders")) is None:
        _ajustes.append(OxmlElement("w:evenAndOddHeaders"))
    return s


def _campo_pagina(p):
    """El número de página como CAMPO de Word, no como número escrito.

    Un número escrito es correcto sólo hasta que el secretario añade un
    párrafo. El campo lo recalcula Word.
    """
    from docx.oxml.ns import qn
    for instr, prop in (("begin", None), (None, "PAGE"), ("end", None)):
        r = p.add_run()._r
        if prop is None:
            f = OxmlElement("w:fldChar")
            f.set(qn("w:fldCharType"), instr)
            r.append(f)
        else:
            t = OxmlElement("w:instrText")
            t.set(qn("xml:space"), "preserve")
            t.text = " PAGE "
            r.append(t)


def _encabezado(doc, texto):
    """El encabezado y el pie, con el formato que se imprime a doble cara.

    ══════════════════════════════════════════════════════════════════════
    LO QUE PEDÍA DAVID Y LO QUE DE VERDAD FALTABA
    ══════════════════════════════════════════════════════════════════════
    David: «me gustaría que el formato, en cuanto a la impresión se refiere,
    sea exactamente igual al de mi engrose. Porque, como verás, cuando hay
    saltos de página, como se imprime por ambas caras, la sangría cambia».

    Lo primero que miré fueron los márgenes, y ahí no estaba: medidos los 30
    engroses de la carpeta, el nuestro ya coincidía con el suyo —21.59 × 34.04,
    izquierda 5, derecha 2, superior e inferior 3— y NINGUNO de los treinta usa
    márgenes en espejo. El documento generado y el suyo tenían la misma caja.

    La diferencia estaba en el encabezado, y es exactamente lo que él describe:

        evenAndOddHeaders ....... 29 de 30 engroses (97%)
        primera página distinta .. 29 de 30 (97%)
        pie con número de página . 27 de 30 (90%)
        ese número, centrado ..... 56 de 58 pies

    Word alterna el encabezado entre página par e impar, y por eso «la sangría
    cambia» al pasar la hoja: el rótulo del expediente se va al canto exterior,
    que en el anverso está a la derecha y en el reverso a la izquierda. El
    documento generado ponía el MISMO encabezado en las tres —y sin número de
    página—, así que impreso por ambas caras no cuadraba con nada y había que
    copiarlo a una hoja con formato, que es justo el trabajo que sobra.

    LA ALINEACIÓN DEL PAR NO LA DECIDE LA MAYORÍA. Emparejando dentro de cada
    documento sale 10 veces derecha→izquierda, 10 veces derecha→centro, 5
    derecha→derecha y 4 centro→centro: empate. Se elige derecha→izquierda
    porque es la única de las cuatro que ALTERNA, y alternar es lo que él pide.
    """
    for s in doc.sections:
        # LA PRIMERA PÁGINA NO LLEVA ENCABEZADO —22 de los 29 la dejan vacía—:
        # ahí va la carátula, y un rótulo encima estorba.
        s.different_first_page_header_footer = True
        # LA PRIMERA PÁGINA, VACÍA Y DECLARADA. Si no se tocan, python-docx no
        # escribe esas dos partes y Word decide por su cuenta qué enseñar. Se
        # crean en blanco, que es como están en el engrose: 22 de los 29 llevan
        # el encabezado de la primera página vacío, y su pie tampoco numera.
        for _primera in (s.first_page_header, s.first_page_footer):
            if _primera is None:
                continue
            _pp = (_primera.paragraphs[0] if _primera.paragraphs
                   else _primera.add_paragraph())
            _pp.text = ""
            _pp.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for parte, alineacion in ((s.header, WD_ALIGN_PARAGRAPH.RIGHT),
                                  (s.even_page_header, WD_ALIGN_PARAGRAPH.LEFT)):
            if parte is None:
                continue
            p = parte.paragraphs[0] if parte.paragraphs else parte.add_paragraph()
            p.text = ""
            r = p.add_run(texto)
            r.bold = True
            r.font.name = FUENTE
            r.font.size = Pt(11)
            r.font.color.rgb = RGBColor.from_string(GRIS_CABECERA)
            p.paragraph_format.alignment = alineacion
        for pie in (s.footer, s.even_page_footer):
            if pie is None:
                continue
            p = pie.paragraphs[0] if pie.paragraphs else pie.add_paragraph()
            p.text = ""
            p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
            _campo_pagina(p)
            for r in p.runs:
                r.font.name = FUENTE
                r.font.size = Pt(11)
                r.font.color.rgb = RGBColor.from_string(GRIS_CABECERA)


# LOS NOMBRES DE PILA FEMENINOS MÁS COMUNES EN EL PODER JUDICIAL no son una
# lista que quepa aquí, y adivinar el género de una persona por su nombre es
# justo lo que este proyecto no hace. Se mira la terminación, que en español
# acierta en la inmensa mayoría, y en la duda se escribe el masculino
# genérico, que es lo que hoy sale y nadie ha objetado.
def _rotulo_secretario(nombre: str) -> str:
    pila = (nombre or "").strip().split()
    if pila and pila[0].lower().rstrip(".").endswith(("a", "triz")) \
            and pila[0].lower() not in ("josé", "jose", "juan", "luca", "elias"):
        return "SECRETARIA"
    return "SECRETARIO"


# EL PROEMIO ES UNA FÓRMULA, NO UNA REDACCIÓN. Medido en 43 de 44 engroses del
# tribunal y en el adelanto que David ajustó: «Querétaro, Querétaro. Resolución
# del [tribunal], correspondiente a la sesión de [fecha].» El modelo escribía
# «Santiago de Querétaro, Querétaro, a ___, el Tercer Tribunal Colegiado…, en
# sesión, emite la presente resolución», que dice lo mismo peor y además
# antepone el «Santiago de» que el corpus no usa en el proemio.
#
# LA FECHA VA EN HUECO porque es la de la sesión, y ésa no existe cuando se
# redacta el proyecto: la fijan los magistrados al revisarlo.
# Corto, en versales y con dos puntos al final: eso es un rótulo de bloque.
_ES_ROTULO_BLOQUE = re.compile(r"^[A-ZÁÉÍÓÚÑ0-9 .,()/-]{4,60}:\s*$")


def _cita_con_rubro(doc, texto: str):
    """La cita, con el RUBRO en negrita, como lo marca David.

    El rubro va entre comillas y en versales; es lo que se busca al hojear y lo
    único que el lector necesita para reconocer el criterio sin leer la frase
    que lo introduce.
    """
    m = re.search(r"[«“\"]([^»”\"]{20,})[»”\"]", texto)
    p = doc.add_paragraph()
    if not m:
        r = p.add_run(texto)
        r.font.name = FUENTE
        r.font.size = TAMANO_CITA
        return _fmt(p, sangria=True, tamano=TAMANO_CITA)
    for trozo, negrita in ((texto[:m.start(1)], False),
                           (m.group(1), True),
                           (texto[m.end(1):], False)):
        if not trozo:
            continue
        r = p.add_run(trozo)
        r.bold = negrita
        r.font.name = FUENTE
        r.font.size = TAMANO_CITA
    return _fmt(p, sangria=True, tamano=TAMANO_CITA)


# ── EL NOMBRE DEL TRIBUNAL NO VA EN VERSALES EN LA PROSA ────────────────────
# David, 13-sep-2026: «el Tribunal va en mayúsculas cuando debe estar con
# mayúscula en cada palabra (del Tribunal Colegiado de Ciudad de México)».
#
# El proemio interpolaba `datos["tribunal"]` tal cual, y ese campo lo teclea el
# secretario en la ficha —casi siempre TODO EN MAYÚSCULAS, porque así aparece en
# la carátula—. En la carátula está bien: es un rótulo. En la prosa del proemio
# no: ahí es el nombre de un órgano y se escribe con inicial en cada palabra.
#
# LAS PARTÍCULAS SE QUEDAN EN MINÚSCULA —«Tribunal Colegiado en Materia Civil
# del Primer Circuito», no «En Materia Civil Del Primer Circuito»—, que es la
# regla del español y lo que hace cualquier engrose del corpus.
_PARTICULAS = {"de", "del", "la", "las", "el", "los", "en", "y", "e", "a"}


# LOS ROMANOS Y LAS SIGLAS SE QUEDAN EN VERSALES (3-oct-2026). El comentario
# del bucle lo prometía —«los ordinales romanos y las siglas cortas se quedan
# como están»— y el código no lo hacía: «SALA REGIONAL DEL CENTRO II» salía
# «Sala Regional del Centro Ii», «XXII CIRCUITO» «Xxii Circuito» e «IMSS»
# «Imss», en la carátula y en los considerandos. Un romano es una palabra hecha
# sólo de I, V, X, L y C que forma un número válido (del I al XCIX); las
# siglas, las federales que aparecen como partes o autoridades.
_RX_ROMANO = re.compile(r"^(?=[IVXLC]+$)(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})$")
_SIGLAS = {"IMSS", "SAT", "TFJA", "TFJFA", "ISSSTE", "INFONAVIT", "FOVISSSTE", "CONAGUA",
           "SHCP", "CDMX", "CFE", "UMA", "INE", "SEP", "SCT", "SICT", "IMPI", "CONDUSEF",
           "CNBV", "SEMARNAT", "PROFEPA", "PROFECO", "FGR", "PGR", "IFT", "CRE", "SADER",
           "SEDATU", "SEDENA", "SEMAR", "SRE", "SEGOB", "SSPC", "INAH", "INEGI", "RAN",
           "SCJN", "PJF", "OAJ", "CJF", "UNAM", "IPN", "PEMEX", "LICONSA", "DICONSA",
           "INDEP", "SAE", "ASF", "INAI", "SFP", "CONAPESCA", "SENASICA", "COFEPRIS",
           "TSJ", "TJA", "JFCA", "JLCA", "CFCRL", "UIF", "CNDH"}


def _palabra_que_no_se_toca(w: str) -> bool:
    """¿Un romano, una sigla o algo con puntos o cifras («S.A.», «22o.»)?"""
    b = w.strip(".,;:()«»\"'")
    if not b:
        return False
    if _RX_ROMANO.match(b) or b in _SIGLAS:
        return True
    return "." in b.rstrip(".") or any(ch.isdigit() for ch in b)


def _nombre_de_organo(t: str) -> str:
    """El nombre del órgano en prosa: inicial en cada palabra, salvo partículas.

    NO SE TOCA LO QUE YA VIENE BIEN ESCRITO. Si el secretario escribió el nombre
    con su capitalización, pasarlo por aquí lo estropearía —«CDMX» acabaría como
    «Cdmx»—. Sólo se rehace lo que llega enteramente en versales, que es la
    marca inequívoca de que viene del rótulo de la carátula. Y dentro de las
    versales, los romanos y las siglas se quedan como están.
    """
    t = " ".join(str(t or "").split()).rstrip(" .,")
    if not t or t != t.upper():
        return t
    fuera = []
    for i, w in enumerate(t.split(" ")):
        b = w.lower()
        if _palabra_que_no_se_toca(w):
            fuera.append(w)
        elif i and b in _PARTICULAS:
            fuera.append(b)
        else:
            fuera.append(b[:1].upper() + b[1:])
    return " ".join(fuera)


def _apertura_compuesta(datos: dict) -> str:
    ciudad = " ".join(str(datos.get("ciudad") or "").split()).rstrip(" .,")
    trib = _nombre_de_organo(datos.get("tribunal"))
    if not (ciudad and trib):
        return ""
    # LA FECHA SIGUE EN HUECO, y es lo correcto: es la de la SESIÓN, que no
    # existe cuando se redacta el proyecto —la fijan los magistrados al
    # revisarlo—. Rellenarla sería inventar un dato de la actuación.
    return (f"{ciudad}. Resolución del {trib}, correspondiente a la sesión de "
            f"{HUECO}.")


def _airear(doc) -> int:
    """Un renglón en blanco entre párrafos, como en el adelanto de David.

    EN UNA SOLA PASADA AL FINAL, no en los cuarenta sitios que escriben
    párrafos: así el aire es una decisión de formato tomada en un lugar, y el
    día que se quiera cambiar —o quitar— se cambia aquí.

    NO SE AIREA TODO. Se salta:
      · lo que ya va seguido de un vacío —la carátula lo pone ella misma—;
      · el último párrafo, que no separa de nada;
      · las firmas, que van juntas por pares: el cargo pegado a su nombre, que
        es como se leen y como él las tiene.
    """
    from docx.shared import Pt as _Pt
    cuerpo = doc.element.body
    puestos = 0
    parrafos = list(doc.paragraphs)
    _firmas = {"MAGISTRADO PONENTE", "SECRETARIO DE TRIBUNAL",
               "SECRETARIA DE TRIBUNAL", "SECRETARIA/O DE TRIBUNAL"}
    for i, p in enumerate(parrafos):
        if not p.text.strip():
            continue
        if i + 1 >= len(parrafos):
            continue
        if not parrafos[i + 1].text.strip():
            continue                      # ya tiene su aire
        if p.text.strip().upper().rstrip(".") in _firmas:
            continue                      # el cargo va pegado a su nombre
        nuevo = copy.deepcopy(p._p)
        for hijo in list(nuevo):
            if hijo.tag.endswith("}r") or hijo.tag.endswith("}hyperlink"):
                nuevo.remove(hijo)
        p._p.addnext(nuevo)
        puestos += 1
    return puestos


# La residencia que SEÑALA y no nombra («, con residencia en esta ciudad»),
# hasta el final del renglón. Ver `_caratula._limpia`.
_RX_RESIDENCIA_QUE_SENALA = re.compile(
    r"\s*,?\s*con\s+(?:residencia|sede|domicilio)\s+en\s+(?:est[ae]|dich[oa]|la\s+misma|el\s+mismo)\s+"
    r"(?:ciudad|capital|localidad|lugar|municipio|entidad|plaza|poblaci[óo]n)\b.*$", re.I)


def _caratula(doc, datos, tipo_asunto: str = "", ponente: tuple = None,
              nuevo: bool = False, relacionados: str = "") -> list:
    """La ficha de identificación. Del asunto, no de ningún otro.

    `ponente` = (rótulo, nombre) ya decidido con la ficha (`_ponente_de_caratula`);
    sin él, el renglón de siempre. `nuevo`: camino nuevo (bandera y ficha).
    `relacionados`: el renglón «RELACIONADO CON…» (`tipos_asunto.rotulo_relacionados`),
    que va justo debajo del encabezado; «» sin asuntos relacionados."""
    # LAS FIGURAS SON DEL TIPO. Esta lista tenía las tres del amparo directo
    # escritas a mano y se imprimía igual en los cuatro, teniendo el tipo en la
    # mano: una revisión rotulaba «QUEJOSO» a quien es recurrente y
    # «AUTORIDAD RESPONSABLE» al Juez de Distrito, que no es parte del recurso
    # sino el órgano de control cuya sentencia se revisa.
    import tipos_asunto as _ta_c
    _t = str(datos.get("tipo_asunto") or tipo_asunto or "amparo_directo")
    # LA PRIMERA LÍNEA NO SE ROTULA. El corpus escribe la clase del asunto como
    # etiqueta —«REVISIÓN FISCAL: 87/2025»—, así que anteponerle «EXPEDIENTE: »
    # la rotula dos veces; salía «EXPEDIENTE: REVISIÓN FISCAL».
    # LOS AVISOS SE DEVUELVEN, NO SE ACUMULAN EN EL MÓDULO. Una lista global
    # la comparten las peticiones que atiende el mismo worker de gunicorn: el
    # aviso de un asunto acabaría en la sentencia de otro. Es la misma clase de
    # error que la de mutar el calendario del cómputo.
    _avisos: list = []
    # UN ENCABEZADO SIN NÚMERO NO IDENTIFICA EL ASUNTO. Salía «REVISIÓN
    # FISCAL» y «RECURSO DE QUEJA CIVIL» a secas, y de ahí el proemio decía
    # «cuyo número consta en autos». Se avisa; no se inventa.
    _enc = str(datos.get("encabezado", ""))
    if _enc and not re.search(r"\b\d{1,5}\s*/\s*\d{2,4}\b", _enc):
        _avisos.append(
            f"EL ENCABEZADO NO TRAE NÚMERO DE EXPEDIENTE: «{_enc[:60]}». Sin él "
            f"el proemio no puede citar el asunto y la carátula no lo "
            f"identifica.")
    campos = [("", _enc)]
    # «RELACIONADO CON…» DEBAJO DEL ENCABEZADO (C6, 3-oct-2026): sólo si el
    # secretario marcó asuntos relacionados. Así lo rotula el banco (AD 456 y
    # 469, Q 24 y 172, RF 4, 21, 26 y 49): la clase y el número, y debajo el
    # asunto con el que se relaciona.
    if str(relacionados or "").strip():
        campos.append(("", str(relacionados).strip()))

    # LA CARÁTULA SE SALTABA LAS DOS NORMALIZACIONES DE LA AUTORIDAD. Escribía
    # `datos["responsable"]` en crudo, y corre aquí —línea 2969— casi
    # trescientas antes de que la 3261 normalice ese mismo dato para el cuerpo
    # y el resolutivo. Con un nombre bueno da igual; con uno tecleado a mano o
    # leído de un encabezado en versales, no: el documento acababa nombrando a
    # la responsable de dos maneras distintas, una en su portada y otra en el
    # punto que resuelve. Se le pasa por el mismo tamiz.
    def _limpia(clave, valor, etiqueta=""):
        # EL CARÁCTER UNA SOLA VEZ (3-oct-2026, AR 208 y 72/2025): con la
        # ficha, si el rótulo ya dice el carácter —«AUTORIDAD RESPONSABLE Y
        # RECURRENTE»— la etiqueta de rol pegada al nombre sobra; con el rótulo
        # a secas («RECURRENTE»), se queda, que es como lo escribe el corpus.
        # «RECURRENTES» (varias autoridades, revisión RF del 3-oct-2026) es el mismo
        # rótulo a secas.
        # Y «(ACTORA)» TRAS LA RECURRENTE ADHESIVA DE LA REVISIÓN FISCAL es el
        # carácter que el rótulo no dice (`tipos_asunto._filas_rf_concordadas`,
        # RF 4/2025: «RECURRENTE ADHESIVO: ***** (ACTORA)»): se queda.
        _actora_adh = clave == "adherente" and re.search(r"\((?:ACTORA?)\)\s*$", str(valor or ""))
        if (nuevo and clave != "responsable" and not _actora_adh
                and etiqueta.strip().upper() not in ("RECURRENTE", "RECURRENTES")):
            valor = _sin_rol(valor) or valor
        if clave != "responsable":
            return valor
        # «CON RESIDENCIA EN ESTA CIUDAD» NO REMITE A NADA EN UN RENGLÓN DE
        # CARÁTULA (3-oct-2026, quinta ronda; Q 337/2026 del banco: «ÓRGANO QUE
        # DICTÓ EL AUTO RECURRIDO: JUZGADO QUINTO DE DISTRITO … EN EL ESTADO DE
        # QUERÉTARO, CON RESIDENCIA EN ESTA CIUDAD.»). La coletilla viene del
        # auto, donde «esta ciudad» es la del juzgado; en el rubro no hay ciudad
        # a la que apuntar. Sólo la que SEÑALA («esta», «este»): la residencia
        # con nombre —«…, con residencia en Ciudad Obregón»— es parte del nombre
        # oficial de los juzgados foráneos y los distingue. En la prosa del
        # V I S T O y de la competencia se queda, que ahí sí se lee.
        if nuevo and _ta_c.normalizar(_t) == "queja":
            valor = _RX_RESIDENCIA_QUE_SENALA.sub("", str(valor or "")).rstrip(" ,") or valor
        return _sin_articulo(_normalizar_autoridad(str(valor or ""))) or valor

    # ═══ CUANDO RECURRE OTRO QUE EL QUEJOSO, SON DOS RENGLONES ══════════════
    # La plantilla dice «{QUEJOSO_A} Y RECURRENTE» porque casi siempre recurre
    # quien perdió el amparo. En el 711/2025 lo ganó la sociedad y recurrió la
    # UIF: con un solo renglón la carátula decía «QUEJOSA Y RECURRENTE: la
    # UIF», que es falso por partida doble. Si `recurrente` viene aparte, la
    # etiqueta se parte: QUEJOSA con la parte y el recurrente con su CARÁCTER
    # («TERCERA INTERESADA Y RECURRENTE» en el AR 631/2025), que es como lo
    # rotula el corpus; y sin «ÓRGANO RECURRIDO», que el corpus no lleva en la
    # revisión (0 de 478). La regla vive en `tipos_asunto.filas_caratula`, la
    # misma que arma la hoja de datos del prompt.
    _filas_caratula = [(et, _limpia(clave, valor, et))
                       for et, clave, valor in _ta_c.filas_caratula(_t, datos)]
    campos += _filas_caratula
    # «SECRETARIO», no «SECRETARIA/O». La barra es de un formulario, no de una
    # sentencia: el adelanto ajustado firma «SECRETARIO:» y quien firma sabe su
    # propio género. Se concuerda con el nombre cuando se puede.
    # «MAGISTRADA PONENTE» SÓLO SI EL CAMPO LO DICE (3-oct-2026): el género no
    # se adivina por el nombre de pila; si el secretario escribió «Magistrada
    # …», el rótulo concuerda con lo que él escribió.
    _mag = str(datos.get("magistrado", "") or "")
    _rot_mag = ("MAGISTRADA PONENTE" if re.match(r"\s*(?:la\s+)?magistrada\b", _mag, re.I)
                else "MAGISTRADO PONENTE")
    _val_mag = datos.get("magistrado", "")
    # CON LA FICHA, EL PONENTE YA VIENE DECIDIDO (3-oct-2026): el del último
    # returno y el cargo en el rótulo, no repetido en el valor.
    if ponente and len(ponente) == 2:
        _rot_mag, _val_mag = ponente
    campos += [(_rot_mag, _val_mag),
               (_rotulo_secretario(str(datos.get("secretario", ""))),
                datos.get("secretario", ""))]
    # ═══════════════════════════════════════════════════════════════════════
    # LA CARÁTULA VA AL LADO DERECHO, SANGRADA, Y CON AIRE ENTRE RENGLONES
    # ═══════════════════════════════════════════════════════════════════════
    # Medido sobre el adelanto que David ajustó a mano: sangría IZQUIERDA de
    # 6.24 cm, justificado, un renglón en blanco entre cada línea, y punto
    # final en cada una. Lo que había —pegado al margen izquierdo, sin aire y
    # sin punto— parecía el encabezado de un oficio; la ficha del rubro va
    # arriba a la derecha, que es donde el ojo la busca en un engrose.
    #
    # LA SANGRÍA ES ABSOLUTA, NO RELATIVA AL MARGEN. El margen izquierdo de
    # estos documentos es de 5 cm —el del Poder Judicial, para el cosido— y la
    # sangría de 6.24 cm se cuenta DESDE ese margen, así que el rubro arranca
    # a algo más de once centímetros del borde. Es lo que da la impresión de
    # bloque a la derecha sin tener que alinear a la derecha, que partiría las
    # palabras de otra manera.
    for etiqueta, valor in campos:
        if not valor:
            continue
        p = doc.add_paragraph()
        # LA NEGRITA ES DE LA ETIQUETA, NO DEL VALOR. Medido run por run en el
        # suyo: «QUEJOSA Y RECURRENTE:» en negrita y el nombre de la sociedad
        # en redonda. Poniéndolo todo en negrita —como hacía— la ficha entera
        # pesa lo mismo y deja de leerse de un golpe: lo que guía el ojo es el
        # contraste entre el rótulo y el dato, no el grosor de la línea.
        #
        # En la primera línea la etiqueta es la CLASE DEL ASUNTO y el valor, el
        # número, así que se parte por los dos puntos: «AMPARO EN REVISIÓN
        # ADMINISTRATIVO:» pesa y «410/2026» no.
        _txt = str(valor).upper().rstrip(" .")
        if etiqueta:
            r1 = p.add_run(f"{etiqueta}: ")
            r1.bold = True
            r2 = p.add_run(_txt + ".")
            r2.bold = False
        elif _txt.startswith("RELACIONADO CON ") and ":" not in _txt:
            # El renglón de los relacionados (C6): pesa la fórmula, como un
            # rótulo; el asunto y su número, no.
            r1 = p.add_run("RELACIONADO CON ")
            r1.bold = True
            r2 = p.add_run(_txt[len("RELACIONADO CON "):].strip() + ".")
            r2.bold = False
        elif ":" in _txt:
            _cab, _resto = _txt.split(":", 1)
            r1 = p.add_run(_cab + ": ")
            r1.bold = True
            r2 = p.add_run(_resto.strip() + ".")
            r2.bold = False
        else:
            r2 = p.add_run(_txt + ".")
            r2.bold = True
        _fmt(p, sangria=False, interlineado=INTERLINEADO,
             alineacion=WD_ALIGN_PARAGRAPH.JUSTIFY)
        p.paragraph_format.left_indent = SANGRIA_CARATULA
        p.paragraph_format.space_after = Pt(0)
        # EL RENGLÓN EN BLANCO ES UN PÁRRAFO, no un `space_after`: así se
        # comporta igual cuando el secretario edita el .docx en Word, que es lo
        # que va a hacer.
        _b = doc.add_paragraph()
        _fmt(_b, sangria=False, interlineado=INTERLINEADO)
        _b.paragraph_format.space_after = Pt(0)
    return _avisos


def _bloque_sintesis(doc, sintesis: dict) -> bool:
    """La última página: la síntesis con forma de tesis del Semanario.

    Va en hoja aparte porque es lo que la vuelve una PORTADA —se desprende y
    encabeza la carpeta que se circula—, y porque en los 23 proyectos del
    corpus que la llevan está siempre al final, detrás de todo.

    Devuelve False si no hay síntesis: entonces el documento conserva las dos
    firmas de siempre. La regresión que hay que evitar aquí no es una portada
    fea, es un proyecto que se queda sin pie.
    """
    if not (sintesis or {}).get("titulo"):
        return False
    doc.add_page_break()
    parrafo(doc, "SÍNTESIS", sangria=False, negrita=True,
            alineacion=WD_ALIGN_PARAGRAPH.CENTER)
    parrafo(doc, "", sangria=False)
    # EL TÍTULO EN VERSALES Y JUSTIFICADO, como los 23 medidos.
    # LOS RÓTULOS, COMO LOS DEJÓ DAVID. En la revisión 650/2025 corrigió a mano
    # la portada que generamos: antepuso «TEMA:» al rubro y cambió «Criterio
    # jurídico:» por «Propuesta de resolución:». Las dos formas están en su
    # corpus —«TEMA:» en 100 de 1,360 documentos y «Criterio jurídico:» en
    # 92—, así que no es que una fuera errónea; es la que él usa. Y
    # «Propuesta de resolución» dice además lo que la síntesis es: un proyecto
    # propone.
    _tp = doc.add_paragraph()
    _rt = _tp.add_run("TEMA: ")
    _rt.bold = True
    _rt2 = _tp.add_run(sintesis["titulo"].rstrip(".") + ".")
    _rt2.bold = True
    _fmt(_tp, sangria=False)
    for etiqueta, clave in (("Hechos: ", "hechos"),
                            ("Propuesta de resolución: ", "criterio"),
                            ("Justificación: ", "justificacion")):
        if not sintesis.get(clave):
            continue
        parrafo(doc, "", sangria=False)
        _p = doc.add_paragraph()
        _r = _p.add_run(etiqueta)
        _r.bold = True
        _p.add_run(sintesis[clave])
        _fmt(_p, sangria=False)
    return True


# ═══════════════════════════════════════════════════════════════════════════
# «NO CONSTA EL SENTIDO DE LA SENTENCIA RECURRIDA», CUANDO SÍ CONSTA
# ═══════════════════════════════════════════════════════════════════════════
# El prompt le ofrece esa frase al modelo como último recurso —«si de verdad no
# consta en lo que tienes, escríbela y sigue»— y el modelo la usa aunque el
# resolutivo del juzgado esté dos páginas antes en el mismo PDF. Salió en la
# revisión 650/2025 de David y ha vuelto a salir en las TRES comprobaciones
# posteriores, con propuestas distintas cada vez: es el comportamiento normal
# del modelo, no un tropiezo suelto.
#
# Ahora el dato se lee del papel, así que la frase se cambia por la buena. NO
# SE INVENTA NADA: si no se pudo leer, la frase se queda como está, que es lo
# honesto —y el aviso del repliegue ya avisa de que hay que comprobarlo—.
_RX_NO_CONSTA_SENTIDO = re.compile(
    r"en\s+la\s+que\s+no\s+consta\s+el\s+sentido\s+de\s+la\s+sentencia\s+recurrida"
    r"|no\s+consta\s+el\s+sentido\s+de\s+la\s+sentencia\s+recurrida", re.I)

_VERBO_A_QUO = {"concede": "en la que se concedió el amparo",
                "niega": "en la que se negó el amparo",
                "sobresee": "en la que se sobreseyó en el juicio",
                "sobresee_niega": "en la que se sobreseyó respecto de un acto "
                                  "y se negó el amparo por los demás",
                "sobresee_concede": "en la que se sobreseyó respecto de un "
                                    "acto y se concedió el amparo por los demás"}


def _con_el_sentido_del_a_quo(texto: str, resolvio: str) -> str:
    """La frase evasiva, cambiada por lo que dice el resolutivo del juzgado."""
    verbo = _VERBO_A_QUO.get((resolvio or "").strip().lower())
    if not verbo or not _RX_NO_CONSTA_SENTIDO.search(texto or ""):
        return texto or ""
    return _RX_NO_CONSTA_SENTIDO.sub(verbo, texto)


def _sobresee_por_cumplimiento(tipo_asunto: str) -> bool:
    """¿Se sobresee porque la sentencia reclamada se dictó en cumplimiento de
    una ejecutoria que no dejó libertad alguna? (30-sep-2026). Sólo en amparo
    directo y sólo si el secretario lo CONFIRMÓ: quitar el estudio de fondo no
    lo decide una lectura (ver `cumplimiento_ejecutoria`)."""
    try:
        import tipos_asunto as _ta_sc
        if _ta_sc.normalizar(tipo_asunto or "amparo_directo") != "amparo_directo":
            return False
        import cumplimiento_ejecutoria as _ce_sc
        return _ce_sc.sobreseer_confirmado()
    except Exception:
        return False


# ═══ LA DEFINITIVIDAD DEL JUICIO ORAL MERCANTIL (30-sep-2026) ════════════════
# El banco trae, como variante de la competencia del amparo directo, la
# coletilla que el tribunal pega al final de ese párrafo cuando lo reclamado
# resolvió un juicio oral mercantil —«…el numeral 1390 bis, que establece que,
# contra las resoluciones pronunciadas en ese tipo de procedimientos, no
# procederá recurso ordinario alguno»— (4 de 62 engroses). Ningún código la
# usaba: el proyecto de un oral mercantil salía sin decir por qué la sentencia
# es definitiva sin haber pasado por una Sala, que es justo lo que David echó
# en falta en el AD 323/2025.
#
# Se pega SÓLO si las dos cosas constan: la instancia es única
# (`tipos_asunto.unica_instancia`, que exige la bandera y el origen) y los autos
# nombran la vía —«juicio oral mercantil», «vía oral mercantil» o el 1390 Bis—.
# El nombre del juzgado no basta: un juzgado de oralidad mercantil también
# lleva el ejecutivo mercantil oral, que es otra vía y otro precepto.
_RX_VIA_ORAL_MERCANTIL = re.compile(
    r"juicio\s+oral\s+mercantil|v[íi]a\s+oral\s+mercantil|\b1390\s*bis\b", re.I)


def coletilla_oral_mercantil(tipo_asunto: str, datos: dict) -> str:
    """La coletilla del banco para el párrafo de competencia, o «»."""
    try:
        if not _ta.unica_instancia(tipo_asunto):
            return ""
        d = datos or {}
        autos = " ".join(str(d.get(k) or "")[:60000] for k in ("acto", "antecedentes"))
        if not _RX_VIA_ORAL_MERCANTIL.search(autos):
            return ""
        import banco as _bk_c
        for v in (_bk_c.apartado(tipo_asunto, "competencia").get("variantes") or []):
            if v.get("id") == "ad-c1-coletilla-oral-mercantil":
                return " ".join(str(v.get("texto") or "").split())
    except Exception:
        return ""
    return ""


# ═══════════════════════════════════════════════════════════════════════════
# LA PROCEDENCIA POR TIPO: LOS DATOS PRIMERO, LA PROSA DESPUÉS (3-oct-2026)
# ═══════════════════════════════════════════════════════════════════════════
# David: «que el adelanto —que ahora serán resultandos y considerandos de
# procedencia— vayan impecables, disminuyendo el margen de error si el
# secretario introduce el auto de admisión y los datos correctos». Hasta hoy
# los considerandos procesales leían con expresiones regulares la prosa que el
# modelo había escrito en los resultandos: el expediente de la existencia (el
# AD_xxii nombró el TOCA como expediente), la fecha del resolutivo de la
# revisión fiscal (la del auto de Presidencia en el RF_yucatan), el inciso del
# 97. Un error del modelo pasaba a lo que parecía determinista.
#
# Con la bandera `procedencia_por_tipo`, el compositor de resultandos
# (`resultandos_por_tipo.componer`) entrega `datos["procesal"]` —lo que leyó de
# la ficha de trámite— y `datos["tramite"]` —la ficha—. Lo que ahí viene MANDA
# sobre cualquier deducción de la prosa; lo que ahí falta sale en HUECO con el
# aviso que nombra el dato, nunca leído de la prosa. Sin la bandera, o sin
# `procesal`, el camino de siempre.
def _rige_procedencia() -> bool:
    """¿Rige `procedencia_por_tipo`? False si el contexto o la bandera faltan."""
    try:
        import contexto_taller as _ct_p
        _f = getattr(_ct_p, "rige", None)
        return bool(_f("procedencia_por_tipo")) if callable(_f) else False
    except Exception:
        return False


def _valor(x) -> str:
    """Un dato de la ficha en texto limpio; «» si no trae nada (o trae hueco)."""
    v = " ".join(str(x or "").split()).strip()
    return "" if (not v or v.strip("*") == "") else v


# ═══════════════════════════════════════════════════════════════════════════
# LOS NOMBRES DE LA FICHA, EN PROSA (3-oct-2026, tercera ronda)
# ═══════════════════════════════════════════════════════════════════════════
# LA ETIQUETA DE ROL DE LA CARÁTULA NO ES PARTE DEL NOMBRE. Las carátulas del
# circuito escriben «GOBERNADOR DEL ESTADO DE QUERÉTARO (AUTORIDAD
# RESPONSABLE)» o «… DEL ISSSTE (DEMANDADA)», y la ficha la trae pegada al
# nombre: la legitimación salía «interpuesto por el GOBERNADOR … (AUTORIDAD
# RESPONSABLE)» (AR 208 y 72/2025) y «… (demandada), en representación de…»
# (RF 4/2025). Y LAS VERSALES SON DE LA CARÁTULA: en la prosa el nombre va en
# mayúsculas y minúsculas, como lo escribe el compositor en el V I S T O y los
# resultandos —si no, el documento nombra a la misma parte de dos maneras—.
_RX_ROL_FINAL = re.compile(
    r"\s*\(\s*(?:la\s+|el\s+)?(?:autoridad\s+)?(?:responsable|demandad[oa]s?|quejos[oa]s?|"
    r"tercer[oa]s?(?:\s+interesad[oa]s?)?|recurrentes?|actor(?:a|es)?|adherentes?|"
    r"parte\s+(?:quejosa|actora|recurrente|demandada|tercera\s+interesada))"
    r"(?:\s+y\s+recurrentes?)?\s*\)\s*$", re.I)


def _sin_rol(x) -> str:
    """«GOBERNADOR … (AUTORIDAD RESPONSABLE)» → «GOBERNADOR …»."""
    v = " ".join(str(x or "").split()).strip()
    for _ in range(3):
        n = _RX_ROL_FINAL.sub("", v).strip(" ,;")
        if n == v:
            break
        v = n
    return v


# UN SOLO CONVERTIDOR DE NOMBRES A PROSA (3-oct-2026, cuarta ronda, E1). Este
# archivo pasaba las versales por `ficha_tramite.sin_versales` y el compositor
# por la suya: el mismo documento decía «la Sucesión a Bienes de…» en el
# V I S T O y «no ampara ni protege a Sucesión a Bienes de JOSÉ GARCÍA RUIZ» en
# el resolutivo (AD 335/2025), «por Conducto de Su Consejo de Administración» y
# «por Propio Derecho» en la legitimación (AD 552/2024, Q 342/2025), y sólo
# convertía el nombre que llegaba ENTERO en versales. La puerta es
# `resultandos_por_tipo.nombre_en_prosa`: rachas en versales dentro de un
# texto mezclado, fórmulas de representación en minúscula, siglas, romanos,
# «S.A. de C.V.», numeraciones y coletillas de carátula fuera.
_RX_ARTICULO_AL_FRENTE = re.compile(r"^(?:el|la|los|las)\s+", re.I)


def _nombre_en_prosa_rpt(x, autoridad: bool = False, con_articulo: bool = False):
    """`resultandos_por_tipo.nombre_en_prosa(x, autoridad, con_articulo)`, o
    None si esa pieza no está. Sin `con_articulo`, el artículo es el que traía
    el papel, ni más ni menos: se quita el que la pieza añada (la firma vieja
    siempre lo pone) y se repone el del papel si la pieza lo quitó."""
    v = _sin_rol(x)
    if not v:
        return ""
    try:
        import resultandos_por_tipo as _rpt_n
        _f = getattr(_rpt_n, "nombre_en_prosa", None)
    except Exception:
        return None
    if not callable(_f):
        return None
    try:
        n = _f(v, autoridad=autoridad, con_articulo=con_articulo)
    except TypeError:
        try:
            n = _f(v, autoridad=autoridad)
        except Exception:
            return None
    except Exception:
        return None
    n = " ".join(str(n or "").split())
    if not n:
        return None
    # SIN ARTÍCULO SI NO SE PIDIÓ Y EL PAPEL NO LO TRAÍA: quien llama lo pone
    # (`legitimacion_de`, las plantillas). EL QUE TRAE EL PAPEL SE RESPETA:
    # «la Titular…» dice algo que el nombre solo no dice (E8), y la puerta, sin
    # artículo pedido, lo quita.
    if not con_articulo:
        _m_in = _RX_ARTICULO_AL_FRENTE.match(v)
        _m_out = _RX_ARTICULO_AL_FRENTE.match(n)
        if _m_out and not _m_in:
            n = _RX_ARTICULO_AL_FRENTE.sub("", n, count=1)
            n = n[:1].upper() + n[1:]
        elif _m_in and not _m_out:
            n = _m_in.group(0).lower() + n
        elif _m_in and _m_out:
            n = _m_out.group(0).lower() + n[_m_out.end():]
    return n


def _en_prosa(x) -> str:
    """El nombre de una parte para la prosa, SIN el artículo que el papel no
    traiga: la regla de `resultandos_por_tipo.nombre_en_prosa` (E1). Si esa
    pieza no está, la de antes: sin la etiqueta de rol y, si viene todo en
    versales, con `ficha_tramite.sin_versales`."""
    v = _sin_rol(x)
    _n = _nombre_en_prosa_rpt(v) if v else ""
    if _n:
        return " ".join(
            w.replace(w.strip(".,;:()«»\"'"), w.strip(".,;:()«»\"'").upper())
            if w.strip(".,;:()«»\"'").upper() in _SIGLAS else w
            for w in _n.split(" "))
    letras = [c for c in v if c.isalpha()]
    if not letras or not all(c.isupper() for c in letras):
        return v
    try:
        import ficha_tramite as _ft_p
        n = _ft_p.sin_versales(v) or _nombre_de_organo(v)
    except Exception:
        n = _nombre_de_organo(v)
    # LAS SIGLAS SIGUEN EN VERSALES («del ISSSTE», no «del Issste»: RF del banco).
    return " ".join(
        w.replace(w.strip(".,;:()«»\"'"), w.strip(".,;:()«»\"'").upper())
        if w.strip(".,;:()«»\"'").upper() in _SIGLAS else w
        for w in n.split(" "))


def _en_prosa_con_articulo(x, autoridad: bool = False) -> str:
    """El nombre para el resolutivo y la legitimación CON el artículo que le
    toca: la Sucesión, la Comunidad, el Ejido (y la autoridad, como órgano, si
    `autoridad`). Misma puerta (E1); sin ella, `_en_prosa` con el artículo de
    los colectivos de `tipos_asunto`."""
    v = _sin_rol(x)
    if not v:
        return ""
    _n = _nombre_en_prosa_rpt(v, autoridad=autoridad, con_articulo=True)
    if _n:
        return " ".join(
            w.replace(w.strip(".,;:()«»\"'"), w.strip(".,;:()«»\"'").upper())
            if w.strip(".,;:()«»\"'").upper() in _SIGLAS else w
            for w in _n.split(" "))
    n = _en_prosa(v)
    try:
        return (_con_articulo(n) if autoridad else _ta.con_articulo_de_colectivo(n)) or n
    except Exception:
        return n


def _plano_nombre(x) -> str:
    v = unicodedata.normalize("NFD", _sin_rol(x).lower())
    v = "".join(c for c in v if unicodedata.category(c) != "Mn")
    v = re.sub(r"^(?:el|la|los|las)\s+", "", v)
    return " ".join(re.findall(r"[a-zñ0-9]+", v))


def _mismo_nombre(a, b) -> bool:
    """El mismo nombre aunque cambien la caja, las tildes, el artículo o un
    complemento al final («Jefe del Departamento de Pensiones» y «Jefe del
    Departamento de Pensiones de la Delegación…»)."""
    pa, pb = _plano_nombre(a), _plano_nombre(b)
    if not (pa and pb):
        return False
    if pa == pb:
        return True
    corto, largo = sorted((pa, pb), key=len)
    if len(corto.split()) >= 2 and (largo.startswith(corto + " ") or f" {corto} " in f" {largo} "):
        return True
    # LA MISMA CABEZA Y NINGUNA PALABRA AJENA (RF 6/2026 del banco: «Subdelegado
    # de Prestaciones Económicas, Delegación Estatal Querétaro, del ISSSTE»
    # contra «Subdelegado de Prestaciones, Delegación Estatal…»). La cabeza
    # cuenta: «Jefatura de Servicios Jurídicos de la Delegación Estatal» NO es
    # «la Delegación Estatal», aunque la contenga.
    _vacias = {"de", "del", "la", "las", "el", "los", "y", "en", "e", "a", "al", "lo"}
    cc = [w for w in corto.split() if w not in _vacias]
    cl = [w for w in largo.split() if w not in _vacias]
    return len(cc) >= 3 and cc[:2] == cl[:2] and set(cc) <= set(cl)


# ═══ EL PONENTE DE LA CARÁTULA ════════════════════════════════════════════
# LOS OCHO ASUNTOS DE CADA TIPO DEL BANCO TUVIERON RETURNO, y la carátula
# nombró al ponente del TURNO mientras su propio resultando decía que los
# autos se returnaron a otro (AD, AR, Q y RF de oro; tras las readscripciones
# de 2025 el returno es lo normal). Firma quien tiene el último turno o
# returno. Y EL CARGO VA EN EL RÓTULO, NO EN EL VALOR: «MAGISTRADO PONENTE:
# MAGISTRADO J. GUADALUPE TAFOYA HERNÁNDEZ» (RF 26/2025); la secretaria en
# funciones no es «MAGISTRADO PONENTE» (AR 222/2025): su rótulo es «PONENTE».
_RX_EN_FUNCIONES = re.compile(
    r",?\s*\(?\s*(?:(?:el|la)\s+)?secretari[oa]\s+(?:de\s+(?:tribunal|estudio\s+y\s+cuenta)\s+)?"
    r"en\s+funciones\s+de\s+magistrad[oa](?:\s+de\s+circuito)?\s*\)?", re.I)
_RX_CARGO_PONENTE = re.compile(r"^\s*(?:(?:el|la)\s+)?(magistrad[oa])(?:\s+de\s+circuito)?\b\.?\s*", re.I)
# EL TRATAMIENTO NO ES PARTE DEL NOMBRE (3-oct-2026, cuarta ronda; AR 239/2025
# del banco: «ponencia de licenciada Bertha Martínez Vega»). Sin returno, la
# carátula habría dicho «PONENTE: LICENCIADA BERTHA MARTÍNEZ VEGA». El
# tratamiento dice el género de quien lo lleva, pero no su cargo (magistrada o
# secretaria en funciones): no decide el rótulo.
_RX_TRATAMIENTO = re.compile(
    r"^\s*(?:(?:el|la)\s+)?(?:licenciad[oa]|lic\.|maestr[oa]|mtr[oa]\.|doctor[a]?|dr[a]?\.)\s+"
    r"(?:en\s+derecho\s+)?", re.I)


def _ponente_sin_cargo(x) -> tuple:
    """(nombre, cargo): cargo «a» (Magistrada), «o» (Magistrado), «funciones»
    (secretaria/o en funciones) o «» si el texto no lo dice. El cargo sólo se
    toma de lo escrito; nunca del nombre de pila. El tratamiento («licenciada»,
    «Mtra.», «Dr.») sale del nombre y no da el cargo."""
    v = " ".join(str(x or "").split()).strip(" .,;")
    cargo = ""
    if _RX_EN_FUNCIONES.search(v):
        v = _RX_EN_FUNCIONES.sub("", v).strip(" .,;")
        cargo = "funciones"
    for _ in range(2):
        m = _RX_CARGO_PONENTE.match(v)
        if m:
            cargo = cargo or m.group(1)[-1].lower()
            v = v[m.end():].strip(" .,;")
        m_t = _RX_TRATAMIENTO.match(v)
        if m_t:
            v = v[m_t.end():].strip(" .,;")
    return v, cargo


def _cargo_del_titulo(t) -> str:
    """El cargo que el auto de turno o de returno escribió junto al nombre
    (`turno.titulo`, `returno.titulo`, que lee `ficha_tramite.leer_auto`):
    «a» (Magistrada), «o» (Magistrado), «funciones» o «»."""
    v = " ".join(str(t or "").split()).lower()
    if not v:
        return ""
    if re.search(r"\ben\s+funciones\b", v):
        return "funciones"
    m = re.match(r"^(?:(?:el|la)\s+)?magistrad([oa])\b", v)
    return m.group(1) if m else ""


def _ponentes_y_cargos_de_la_ficha(ficha: dict) -> tuple:
    """((ponente del turno, su título), (ponente del ÚLTIMO returno, su
    título)) tal como los trae la ficha. El returno puede venir como un dict o
    como una lista de ellos; el título es el del mismo auto que el nombre."""
    f = ficha if isinstance(ficha, dict) else {}
    tur = f.get("turno") if isinstance(f.get("turno"), dict) else {}
    ret = f.get("returno")
    rets = ret if isinstance(ret, list) else ([ret] if isinstance(ret, dict) else [])
    ult, tit_u = "", ""
    for r_ in rets:
        if isinstance(r_, dict) and _valor(r_.get("ponente")):
            ult, tit_u = _valor(r_.get("ponente")), _valor(r_.get("titulo"))
    return (_valor(tur.get("ponente")), _valor(tur.get("titulo"))), (ult, tit_u)


def _ponentes_de_la_ficha(ficha: dict) -> tuple:
    """(ponente del turno, ponente del ÚLTIMO returno) tal como los trae la
    ficha. El returno puede venir como un dict o como una lista de ellos."""
    (tur, _), (ult, _) = _ponentes_y_cargos_de_la_ficha(ficha)
    return tur, ult


def _ponente_de_caratula(datos: dict, ficha: dict, pro: dict) -> tuple:
    """(rótulo, nombre, avisos) del renglón del ponente, con la ficha.

    Manda el último returno sobre el turno cuando el formulario viene vacío o
    trae al del turno; si el formulario dice OTRA persona, se respeta (es lo
    que el secretario escribió) y se avisa. El cargo, de lo escrito: del
    formulario o, si es la misma persona, del auto de turno o de returno."""
    avisos = []
    mag, car = _ponente_sin_cargo(datos.get("magistrado"))
    _de_la_ficha = False
    (t_raw, tit_t), (r_raw, tit_r) = _ponentes_y_cargos_de_la_ficha(ficha)
    if not (t_raw or r_raw):
        t_raw = _valor((pro or {}).get("ponente"))
    tur, car_t = _ponente_sin_cargo(t_raw)
    ret, car_r = _ponente_sin_cargo(r_raw)
    # EL CARGO QUE DIJO EL AUTO (3-oct-2026, cuarta ronda, E3; Q 342/2025 del
    # banco): «túrnense los autos a la Magistrada Jenica Campos Juárez» deja
    # `turno.titulo` = «Magistrada» y el nombre sin el cargo; se leía sólo el
    # nombre y salía «MAGISTRADO PONENTE» con un aviso que decía, en falso, que
    # el auto no lo dice.
    car_t = car_t or _cargo_del_titulo(tit_t)
    car_r = car_r or _cargo_del_titulo(tit_r)
    # El del compositor (`procesal.ponente_titulo`), para el ponente vigente.
    _tit_p, _pon_p = _valor((pro or {}).get("ponente_titulo")), _valor((pro or {}).get("ponente"))
    if _tit_p and _pon_p:
        if ret and not car_r and _mismo_nombre(ret, _pon_p):
            car_r = _cargo_del_titulo(_tit_p)
        elif not ret and tur and not car_t and _mismo_nombre(tur, _pon_p):
            car_t = _cargo_del_titulo(_tit_p)
    ult, car_u = (ret, car_r) if ret else (tur, car_t)
    if not mag and ult:
        mag, car = ult, car_u
        _de_la_ficha = True
    elif mag and ret and tur and _mismo_nombre(mag, tur) and not _mismo_nombre(mag, ret):
        avisos.append(
            f"LA CARÁTULA TRAÍA AL PONENTE DEL TURNO ({tur}) Y LOS AUTOS SE RETURNARON A {ret}: "
            f"se puso a {ret}, que es quien firma el proyecto. Compruébalo.")
        mag, car = ret, car_r
        _de_la_ficha = True
    elif mag and ult and not _mismo_nombre(mag, ult):
        avisos.append(
            f"EL PONENTE DE LA CARÁTULA ({mag}) NO ES EL DEL "
            f"{'ÚLTIMO RETURNO' if ret else 'TURNO'} DE LA FICHA ({ult}): la carátula debe nombrar "
            f"a quien firma el proyecto. Compruébalo en el auto de "
            f"{'returno' if ret else 'turno'}.")
    if mag and not car:
        for n_, c_ in ((ret, car_r), (tur, car_t)):
            if c_ and _mismo_nombre(mag, n_):
                car = c_
                break
    rot = {"funciones": "PONENTE", "a": "MAGISTRADA PONENTE"}.get(car, "MAGISTRADO PONENTE")
    # EL AVISO DEL RÓTULO POR OMISIÓN, VENGA EL NOMBRE DE DONDE VENGA (3-oct-2026,
    # cuarta ronda, E3). Salía sólo si el nombre lo había puesto la ficha: el
    # tecleado sin cargo («Jenica Campos Juárez») se firmaba «MAGISTRADO
    # PONENTE» sin una palabra (AD 128, 279 y 552; RF 4 y 49 del banco). El
    # género no se adivina por el nombre de pila: si nadie escribió el cargo,
    # el rótulo va por omisión y se dice.
    if mag and not car:
        avisos.append(
            "«MAGISTRADO PONENTE» VA POR OMISIÓN: ni el formulario ni el auto de turno o de "
            "returno dicen el cargo. Si es Magistrada, o secretaria o secretario en funciones, "
            "corrígelo.")
    return rot, mag, avisos


# ═══ LO RECLAMADO EN EL AMPARO DIRECTO: SENTENCIA, RESOLUCIÓN O LAUDO ══════
# (3-oct-2026, AD 349 y 552 del banco). El V I S T O, el resultando y el
# resolutivo decían «la resolución dictada…» y la competencia, la existencia,
# la legitimación y la oportunidad seguían diciendo «sentencia definitiva» y
# «la sentencia reclamada»: estaban fijas en el código. La clase la da la ficha.
def _clase_del_acto_ad(pro: dict, ficha: dict) -> str:
    """«sentencia» | «resolucion» | «laudo» del acto reclamado en el AD."""
    acto = (ficha or {}).get("acto") if isinstance((ficha or {}).get("acto"), dict) else {}
    for c in (_valor((pro or {}).get("clase")), _valor(acto.get("clase"))):
        c = c.lower().replace("ó", "o")
        if c in ("sentencia", "resolucion", "laudo"):
            return c
    _txt = " ".join([_valor((pro or {}).get("acto_reclamado")),
                     _valor((pro or {}).get("descripcion_acto"))]).lower()
    return "laudo" if "laudo" in _txt else "resolucion" if "resoluci" in _txt else "sentencia"


# ═══ LA FRACCIÓN DEL 97 QUE CONSTA, O NINGUNA (3-oct-2026, quinta ronda, F3) ══
# Q 335/2025 del banco: sin la fracción en la ficha, el compositor dejó la vía
# en hueco en el V I S T O, y la competencia y la procedencia suponían la I y el
# amparo indirecto. Consta si la dice el compositor o la ficha (`fraccion_97`),
# o si consta la vía (indirecto ⇒ I; directo ⇒ II). Nada más: el nombre del
# órgano lo pesa el compositor (`resultandos_por_tipo._fraccion_97`), que es
# quien sabe que «del Distrito Judicial» no es un Juzgado de Distrito.
def _fraccion_97_de(pro: dict, ficha: dict) -> str:
    """«I», «II» o «» si no consta."""
    pro = pro if isinstance(pro, dict) else {}
    ficha = ficha if isinstance(ficha, dict) else {}
    for v in (pro.get("fraccion_97"), ficha.get("fraccion_97")):
        f = re.sub(r"(?i)fracci[óo]n|fr\.", "", _valor(v)).strip(" .").upper()
        if f in ("I", "II"):
            return f
    acto = ficha.get("acto") if isinstance(ficha.get("acto"), dict) else {}
    for v in (pro.get("via_amparo"), acto.get("via")):
        v = _valor(v).lower()
        if v == "indirecto":
            return "I"
        if v == "directo":
            return "II"
    return ""


_RX_FR97_I = re.compile(r"(\b97,\s+fracci[óo]n\s+)I(?=\s*,)")
_RX_JUICIO_INDIRECTO = re.compile(r"(\bjuicio\s+de\s+amparo\s+)indirecto\b")


def _fraccion_97_en_hueco(texto: str, huecos: list = None, ident: str = "") -> str:
    """La cadena de la fracción I sin afirmarla: «97, fracción *********,» y
    «juicio de amparo *********». Apunta el dato en `huecos` para su aviso."""
    t = _RX_JUICIO_INDIRECTO.sub(lambda m: m.group(1) + HUECO,
                                 _RX_FR97_I.sub(lambda m: m.group(1) + HUECO, texto or ""))
    if t != (texto or "") and isinstance(huecos, list) and (ident, "fraccion_97") not in huecos:
        huecos.append((ident, "fraccion_97"))
    return t


# ═══ LA FORMA DE NOTIFICACIÓN QUE NADIE DIJO (3-oct-2026, quinta ronda, F1) ═══
# Las dos reglas que llegan solas: «personal», el valor por omisión del
# formulario (`regla_surtimiento: Form("personal")`), y «oficio», la que D2 pone
# a la autoridad que recurre. La ficha marca su fuente («omision»,
# «omision_autoridad»); una ficha sin `forma_notificacion` tampoco la declara.
_FUENTES_FORMA_OMISION = ("omision", "omision_autoridad")


def _forma_de_omision(ficha: dict, computo) -> str:
    """«omision» u «omision_autoridad» si el cómputo corrió con una regla que no
    dijo ningún papel; «» si la forma consta (la declaró el secretario o la leyó
    la ficha del auto, del escrito o del acto) o si la regla es otra."""
    if not isinstance(ficha, dict) or not ficha:
        return ""
    clave = str(getattr(getattr(computo, "regla", None), "clave", "") or "")
    # «cnpcf_personal» (C7, 3-oct-2026): la personal del Código Nacional que el
    # guardián de materia pone en lo agrario de la Ciudad de México en lugar de
    # la de omisión; tampoco es un dato del papel (F1). Ni «cfpc_personal»
    # (integración, 3-oct-2026), la del Código Federal que el desplegable
    # propone fuera de la Ciudad de México: elegir la regla no es leer la forma.
    if clave not in ("personal", "oficio", "cnpcf_personal", "cfpc_personal"):
        return ""
    _fu = ficha.get("fuentes") if isinstance(ficha.get("fuentes"), dict) else {}
    fuente = str(_fu.get("forma_notificacion") or "").strip().lower()
    if fuente in _FUENTES_FORMA_OMISION:
        return fuente
    if _valor(ficha.get("forma_notificacion")):
        return ""
    return "omision_autoridad" if clave == "oficio" else "omision"


def _sin_forma_de_notificacion(texto: str, computo) -> str:
    """«…el seis de octubre de dos mil veinticinco de manera personal y surtió
    efectos…» → «…el seis de octubre de dos mil veinticinco y surtió efectos…».
    Sólo la forma de la regla del cómputo y sólo en ese sitio."""
    desc = str(getattr(getattr(computo, "regla", None), "descripcion", "") or "").strip()
    if not desc:
        return texto or ""
    return re.sub(r"\s+" + re.escape(desc) + r"(?=\s+y\s+surti[óo]\s+efectos\b)", "",
                  texto or "", count=1)


def _sin_forma_de_aviso(texto: str, computo) -> str:
    """«…que la notificación de manera personal surtió efectos…» → «…que la
    notificación surtió efectos…» en el aviso del precepto (F1)."""
    desc = str(getattr(getattr(computo, "regla", None), "descripcion", "") or "").strip()
    if not desc:
        return texto or ""
    return re.sub(r"(\bla\s+notificaci[óo]n)\s+" + re.escape(desc) + r"(?=\s+surti[óo]\b)", r"\1",
                  texto or "")


def _aviso_forma_omision(computo, tipo: str, papel: str = "") -> str:
    """El aviso de F1, el de `fase0_oportunidad.aviso_forma_no_consta` si está."""
    try:
        import fase0_oportunidad as _f0_af
        _f = getattr(_f0_af, "aviso_forma_no_consta", None)
        if callable(_f):
            _a = _f(computo, tipo, papel)
            if _a:
                return str(_a)
    except Exception:
        pass
    clave = str(getattr(getattr(computo, "regla", None), "clave", "") or "")
    if clave == "oficio":
        como = "por oficio (artículo 31, fracción I, de la Ley de Amparo)"
    elif _ta.normalizar(tipo) == "amparo_directo":
        como = "personal (la regla de la ley que rige el acto)"
    else:
        como = "personal (artículo 31, fracción II, de la Ley de Amparo)"
    return ("LA FORMA DE NOTIFICACIÓN NO CONSTA (forma_notificacion): ningún papel dice cómo se "
            f"notificó y el cómputo la contó {como}. El considerando no la afirma: dice la fecha "
            "y cuándo surtió efectos. Compruébala en la constancia de notificación.")


# ═══ LOS ASUNTOS RELACIONADOS QUE MARCÓ EL SECRETARIO (C6, 3-oct-2026) ═══════
# David: «siempre y cuando haya asuntos relacionados. No vamos a meter conexidad
# en automático. Hay que habilitar en el taller la opción de con un clic
# precisar si existen asuntos relacionados y con ello se genera el
# considerando». Sólo la lista que el secretario marcó en «Trámite en este
# tribunal» (`procesal.relacionados`, o la de la ficha); nunca se lee de los
# papeles. Con ella, el renglón del rubro («RELACIONADO CON…») y el considerando
# de conexidad o de hecho notorio (`tipos_asunto.considerando_relacionados`).
_MATERIA_DEL_ENCABEZADO = (("civil", "civil"), ("administrativ", "administrativa"),
                           ("mercantil", "mercantil"), ("laboral", "laboral"),
                           ("familiar", "familiar"), ("agrari", "agraria"), ("penal", "penal"))


def _relacionados_del_asunto(tipo: str, datos: dict, pro: dict, ficha: dict) -> tuple:
    """(lista, materia, número del asunto). La lista, normalizada
    (`tipos_asunto.relacionados_validos`) y sin el propio asunto; vacía si el
    secretario no marcó ninguno. La materia es la del asunto (con ella se
    nombran los relacionados: «amparo directo civil 452/2025»)."""
    pro = pro if isinstance(pro, dict) else {}
    ficha = ficha if isinstance(ficha, dict) else {}
    crudo = pro.get("relacionados")
    if not (isinstance(crudo, list) and crudo):
        crudo = ficha.get("relacionados")
    try:
        lista = _ta.relacionados_validos(crudo)
    except Exception:
        lista = []
    if not lista:
        return [], "", ""
    numero = (_valor(ficha.get("numero")) or _valor(datos.get("numero"))
              or str(datos.get("encabezado") or ""))
    _m = re.search(r"(\d{1,6})\s*/\s*(\d{4})", numero or "")
    numero = f"{int(_m.group(1))}/{_m.group(2)}" if _m else ""
    _t = _ta.normalizar(tipo)
    lista = [r for r in lista if not (numero and r["tipo"] == _t and r["numero"] == numero)]
    materia = (_valor(ficha.get("materia")) or _valor(pro.get("materia"))
               or str(datos.get("materia") or "").strip()).lower()
    if not materia:
        _enc = str(datos.get("encabezado") or "").lower()
        materia = next((cl for m_, cl in _MATERIA_DEL_ENCABEZADO if m_ in _enc), "")
    # LA MISMA MATERIA QUE EL V I S T O (integración, 3-oct-2026). El compositor
    # nombra a los relacionados con la clave de la materia (lo mercantil y lo
    # familiar como civil, lo agrario como administrativo; «materias
    # administrativa y civil», que es la del tribunal, sin palabra) y, en la
    # revisión fiscal sin materia, con la administrativa; el rubro y el
    # considerando decían otra cosa con la materia cruda («amparo directo
    # 456/2025» en el rubro de la RF y «…administrativo 456/2025» en su V I S T O).
    # La concordancia con cada tipo la hace `tipos_asunto` (la del encabezado).
    try:
        import resultandos_por_tipo as _rpt_m
        _mc = _rpt_m._materia_concordada(_rpt_m._materia_clave(materia), "f")
        materia = _mc or ("administrativa" if _t == "revision_fiscal" else "")
    except Exception:
        if not materia and _t == "revision_fiscal":
            materia = "administrativa"
    return lista, materia, numero


_RX_PONE_FIN = re.compile(
    r"desech|sobrese|caducidad|caduc[óo]|perenci[óo]n|incompeten|"
    r"(?:pus[oa]|pone|ponga|pusiera)\s+fin\s+al\s+(?:juicio|procedimiento)|"
    r"(?:den|dio|da)\s+por\s+concluid", re.I)


def _pone_fin_al_juicio(pro: dict, ficha: dict) -> str:
    """Lo que la ficha dice del sentido de lo reclamado si es un desechamiento,
    un sobreseimiento, una caducidad o una incompetencia (o lo dice ella misma:
    «puso fin al juicio»), entre comillas; «» si no. Decide si la resolución
    reclamada en el AD «puso fin al juicio» (art. 170, fr. I, LA; E4)."""
    acto = (ficha or {}).get("acto") if isinstance((ficha or {}).get("acto"), dict) else {}
    for v in (_valor(acto.get("sentido")), _valor(acto.get("resolvio")),
              _valor(acto.get("sentido_clave")), _valor(acto.get("resolvio_mixto")),
              _valor((pro or {}).get("sentido")), _valor((pro or {}).get("resolvio")),
              _valor(acto.get("clase"))):
        if v and _RX_PONE_FIN.search(v):
            return "«" + (v if len(v) <= 140 else v[:140].rsplit(" ", 1)[0] + "…") + "»"
    return ""


# ═══ LA UNIDAD QUE RECURRE EN LA REVISIÓN FISCAL ═══════════════════════════
# «X, en representación de Y» trae a las dos: la unidad que firma el oficio y
# la autoridad demandada cuya defensa lleva (RF 2, 7 y 6/2025). El compositor
# las da por separado (`recurrente_unidad`, `autoridad_demandada`); si no las
# dio, se parten aquí con la misma regla.
_RX_EN_REPRESENTACION = re.compile(
    r",?\s+(?:en\s+representaci[óo]n|en\s+nombre|por\s+conducto)\s+(?:de\s+(?:la|las|los)\s+|del?\s+)", re.I)


def _partir_representacion(x) -> tuple:
    """(unidad, representada) de «X, en representación de Y»; (X, «») si no."""
    v = _sin_rol(x)
    m = _RX_EN_REPRESENTACION.search(v)
    if not m:
        return v, ""
    return v[:m.start()].strip(" ,;"), _sin_rol(v[m.end():]).strip(" ,;.")


# ═══ LA DELEGACIÓN DE LA SUPREMA CORTE EN EL AMPARO EN REVISIÓN ════════════
# (3-oct-2026, AR 201, 208 y 72/2025 del banco). Si en el amparo indirecto se
# impugnó una norma general y el problema subsiste en la revisión, este
# tribunal conoce por delegación de la SCJN (art. 83, segundo párrafo, LA; AG
# 2/2025 (12a.) y 11/2025 (12a.)), la cadena de la competencia es otra y el
# proyecto se hace público tres días antes de la lista (art. 73, segundo
# párrafo, LA: «… deberán hacer públicos los proyectos de sentencias… cuando
# menos con tres días de anticipación»). Señal de la ficha: entre las
# autoridades de la demanda está el órgano que expide la ley —Congreso,
# Legislatura, Cámara—, o un acto reclamado es una ley, un artículo de una ley
# o un decreto. No se cambia la competencia: se avisa, porque decidir si el
# problema subsiste es del estudio.
_RX_LEGISLADOR = re.compile(r"\b(?:congreso|legislatura|c[áa]mara\s+de\s+(?:diputad[oa]s|senador[ea]s)|"
                            r"asamblea\s+legislativa)\b", re.I)
_RX_ACTO_NORMA = re.compile(
    r"^\W*(?:(?:la|el|los|las)\s+)?(?:(?:expedici[óo]n|promulgaci[óo]n|aprobaci[óo]n|publicaci[óo]n|"
    r"refrendo)\b[^.]{0,80}?\b(?:de\s+)?(?:la|el|los)?\s*)?(?:ley|c[óo]digo|reglamento|decreto|"
    r"art[íi]culos?\s+\d+[^.]{0,80}?\b(?:de\s+la\s+ley|del\s+c[óo]digo|del\s+reglamento|"
    r"del\s+decreto|de\s+la\s+constituci[óo]n))\b", re.I)


def _aviso_delegacion_scjn(autoridades: list, actos: list, avisos_previos: list,
                           resolvio: str = "", papel: str = "", clase: str = "") -> list:
    """[] o [aviso] sobre la delegación de la SCJN en el amparo en revisión."""
    if clase in ("interlocutoria_suspension", "auto_sobreseimiento", "reposicion_constancias"):
        return []
    leg = [a for a in (autoridades or []) if _RX_LEGISLADOR.search(a or "")]
    normas = [a for a in (actos or []) if _RX_ACTO_NORMA.search(a or "")]
    ya = any("CONSTITUCIONALIDAD DE UNA NORMA" in str(a) for a in (avisos_previos or []))
    if not (leg or normas or ya):
        return []
    r = (resolvio or "").strip().lower()
    concede = "concede" in r
    aut = (papel or "").strip().lower() == "autoridad"
    if ya and not (concede and aut):
        return []
    por = []
    if leg:
        por.append("entre las autoridades de la demanda está " + "; ".join(leg[:2]))
    if normas:
        _n0 = normas[0] if len(normas[0]) <= 90 else normas[0][:90].rsplit(" ", 1)[0].rstrip(" ,;") + "…"
        por.append("se reclamó «" + _n0 + "»")
    texto = ("LA DEMANDA DE AMPARO IMPUGNÓ UNA NORMA GENERAL (" + "; ".join(por) + "). " if por
             else "SE IMPUGNÓ UNA NORMA GENERAL. ")
    if concede and aut:
        texto += ("El juzgado concedió el amparo y recurre la autoridad: el problema de "
                  "constitucionalidad MUY PROBABLEMENTE SUBSISTE y este tribunal conocería por "
                  "DELEGACIÓN de la SCJN (artículo 83, segundo párrafo, de la Ley de Amparo; punto "
                  "cuarto, fracción I, inciso que corresponda, del Acuerdo General 2/2025 (12a.), en "
                  "relación con el punto segundo del 11/2025 (12a.), ambos del Pleno de la SCJN). ")
    else:
        texto += ("Si el problema de constitucionalidad subsiste en la revisión, este tribunal "
                  "conoce por DELEGACIÓN de la SCJN (artículo 83, segundo párrafo, de la Ley de "
                  "Amparo; Acuerdos Generales 2/2025 (12a.) y 11/2025 (12a.) del Pleno de la SCJN) "
                  "y la competencia cambia. ")
    texto += ("La competencia se escribió con la cadena ordinaria: compruébalo. Y si conoce por "
              "delegación, el proyecto se hace público al menos tres días antes de la lista "
              "(artículo 73, segundo párrafo, de la Ley de Amparo).")
    return [texto]


# EL 5o., CUARTO PÁRRAFO, DE LA LFPCA: «La representación de las autoridades
# corresponderá a las unidades administrativas encargadas de su defensa
# jurídica» (texto local). Es la norma que hace parte legítima a la unidad que
# firma el oficio; el 63, párrafo primero, sólo dice que la revisión la
# interpone esa unidad. Los engroses del banco lo citan (RF 21, 26 y 49/2025).
_RX_63_PRIMERO = re.compile(
    r"(art[íi]culo\s+63,\s+p[áa]rrafo\s+primero),(\s+de\s+la\s+Ley\s+Federal\s+de\s+Procedimiento\s+"
    r"Contencioso\s+Administrativo)")


def _con_el_5o_lfpca(texto: str) -> str:
    if not texto or re.search(r"5o\.,\s+cuarto\s+p[áa]rrafo", texto):
        return texto
    return _RX_63_PRIMERO.sub(r"\1, en relación con el 5o., cuarto párrafo,\2", texto, count=1)


# ═══ LA LEGITIMACIÓN, CON LO QUE LA FIRMA ADMITA (3-oct-2026, cuarta ronda) ══
# `tipos_asunto.legitimacion_de` crece en cada ronda (`avisos`, `plural`, la
# norma impugnada): se le pasan sólo los argumentos que su firma conoce, para
# que una pieza a medio actualizar no tumbe la legitimación entera.
def _admite(f, nombre: str) -> bool:
    """¿La firma de `f` admite el argumento `nombre`? (True si no se sabe)."""
    try:
        import inspect as _insp
        _params = _insp.signature(f).parameters
    except (TypeError, ValueError):
        return True
    return nombre in _params or any(p_.kind == p_.VAR_KEYWORD for p_ in _params.values())


def _legitimacion_de(t: str, parte: str, rep: str, **kw) -> str:
    """`tipos_asunto.legitimacion_de(t, parte, rep, HUECO, **kw)` sin los
    argumentos con nombre que su firma no admite. Los que valen None no van."""
    f = _ta.legitimacion_de
    return f(t, parte, rep, HUECO, **{k: v for k, v in kw.items()
                                      if v is not None and _admite(f, k)})


# UN VALOR CORTADO NO SE FIRMA (3-oct-2026, cuarta ronda, E8; RF 2/2025 y
# 6/2026 del banco: «por conducto de su Titular de la Unidad Jurídica…»). Se
# deja fuera de la legitimación con su aviso; el del compositor sobre el mismo
# campo se junta con éste (`_un_aviso_por_hecho`).
_RX_TRUNCADO = re.compile(r"(?:…|\.\.\.)\s*$")


def _sin_truncar(x, clave: str = "", avisos: list = None, de_donde: str = "escrito") -> str:
    """El valor, o «» si viene cortado con «…»; entonces, si se da `avisos`,
    el aviso con la forma del de `tipos_asunto` («DATO TRUNCADO (clave = …)»),
    para que el de las demás piezas sobre el mismo campo se junte con él."""
    v = _valor(x)
    if not _RX_TRUNCADO.search(v):
        return v
    if isinstance(avisos, list) and clave:
        avisos.append(
            f"DATO TRUNCADO ({clave} = «{v}»): termina en puntos suspensivos y no se escribió "
            f"en la legitimación. Cópialo completo del {de_donde} de interposición.")
    return ""


# «LA TITULAR» ES DEL PAPEL Y SE CONSERVA (3-oct-2026, cuarta ronda, E8; RF 4,
# 6, 21 y 49/2025 del banco): el compositor la escribía en el resultando y la
# legitimación, que recibía la unidad sin artículo, volvía a «el Titular». Sin
# artículo en el papel, «el Titular» va por omisión y con aviso.
# Y NO SÓLO «LA TITULAR» (3-oct-2026, quinta ronda, F5; AD 128/2025 del banco:
# «la Oficial Mayor y Coordinadora…» salía «el Oficial Mayor»): cualquier cargo
# que se escribe igual en los dos géneros —Titular, Oficial, Fiscal, Agente,
# Representante, Encargado/a— conserva el artículo que trae el papel. El aviso
# del masculino por omisión sigue siendo el de «Titular» (`_RX_TITULAR`).
_CARGO_DE_DOS_GENEROS = r"(?:titular|oficial|fiscal|agente|representante|encargad[oa])"
_RX_ART_TITULAR = re.compile(r"^\s*(el|la)\s+" + _CARGO_DE_DOS_GENEROS + r"\b", re.I)
_RX_TITULAR = re.compile(r"^\s*titular\b", re.I)
_RX_CARGO_DE_DOS_GENEROS = re.compile(r"^\s*" + _CARGO_DE_DOS_GENEROS + r"\b", re.I)


def _con_la_titular(nombre: str, crudo: str) -> str:
    """«Titular de la X» → «la Titular de la X» (o «el…») si el papel
    (`crudo`) lo dice; si no, como viene. Igual con «Oficial», «Fiscal»,
    «Agente», «Representante» y «Encargado/a» (F5)."""
    m = _RX_ART_TITULAR.match(crudo or "")
    if nombre and m and _RX_CARGO_DE_DOS_GENEROS.match(nombre):
        return f"{m.group(1).lower()} {nombre.strip()}"
    return nombre


def _legitimacion_con_la_ficha(tipo: str, ficha: dict, pro: dict, datos: dict,
                               papel_op: str = "", autoridad_demandada: str = "") -> tuple:
    """(párrafo de legitimación, avisos) con quien promueve o recurre según la
    ficha: su nombre en prosa y sin etiqueta de rol, su representante y su
    figura, y su carácter. («», avisos) si la ficha no basta: queda el de
    siempre. La forma la da `tipos_asunto.legitimacion_de`.

    CUARTA RONDA (3-oct-2026): el nombre por la misma puerta que el V I S T O
    (`resultandos_por_tipo.nombre_en_prosa`, E1); el plural que decidió el
    compositor (`procesal.plural`, E2); la autoridad que recurre, como órgano y
    con su artículo; «la Titular» del papel (E8); los valores cortados con
    «…», fuera."""
    t = _ta.normalizar(tipo)
    f = ficha if isinstance(ficha, dict) else {}
    pro = pro if isinstance(pro, dict) else {}
    avisos: list = []
    _de_donde = "oficio" if t == "revision_fiscal" else "escrito"
    rep = _en_prosa(_sin_truncar(f.get("representante"), "representante", avisos, _de_donde))
    fig = _sin_truncar(f.get("figura_representante"), "figura_representante", avisos, _de_donde)
    # EL NÚMERO LO DECIDE EL COMPOSITOR (E2): `plural` es el de quien promueve
    # o recurre —la parte de esta legitimación en el AR y la queja—;
    # `plural_quejoso`, el de la quejosa —la del amparo directo—.
    _pl_rec = pro.get("plural") if isinstance(pro.get("plural"), bool) else None
    _pl_q = pro.get("plural_quejoso") if isinstance(pro.get("plural_quejoso"), bool) else _pl_rec
    # EL GÉNERO, SÓLO SI EL PAPEL LO DICE (`genero_quejoso`/`genero_recurrente`,
    # que lee `tipos_asunto.genero_en_el_papel`): nunca del nombre de pila.
    _g_q = _valor(datos.get("genero_quejoso")).lower()[:1]
    _g_r = _valor(datos.get("genero_recurrente")).lower()[:1]
    if t == "amparo_directo":
        parte = _en_prosa(datos.get("quejoso"))
        rep = _en_prosa(_sin_truncar(datos.get("representante"), "representante", avisos)) or rep
        fig = _sin_truncar(datos.get("figura_representante"), "figura_representante", avisos) or fig
        if rep and _mismo_nombre(rep, parte):
            rep, fig = "", ""
        if not parte:
            return "", avisos
        _av_b: list = []
        texto = _legitimacion_de(t, parte, rep, figura=fig, moral=datos.get("quejoso_moral"),
                                 plural=_pl_q, genero=_g_q or None, avisos=_av_b)
        avisos.extend(a_ for a_ in _av_b if a_ not in avisos)
        return texto, avisos
    if t == "revision_fiscal":
        _crudo = _valor(f.get("promovente")) or _valor(datos.get("quejoso"))
        # «LA TITULAR» COMO LA ESCRIBIÓ EL COMPOSITOR (E8): su artículo es dato
        # del papel (`recurrente_unidad_con_articulo`, `recurrente_la_titular`).
        # Los demás artículos los pone `legitimacion_de`, como siempre.
        _con_art_c = _valor(pro.get("recurrente_unidad_con_articulo"))
        _art_papel = (_con_art_c if _RX_ART_TITULAR.match(_con_art_c) else
                      "la titular" if (pro.get("recurrente_la_titular") is True
                                       and not _RX_ART_TITULAR.match(_crudo)) else _crudo)
        unidad = (_valor(pro.get("recurrente_unidad"))
                  or _RX_ARTICULO_AL_FRENTE.sub("", _con_art_c, count=1))
        # Y QUIEN LA FIRMÓ, si es su titular: «lo hizo valer {nombre}, {unidad}».
        rep = rep or _en_prosa(_sin_truncar(pro.get("recurrente_nombre")))
        dem = _valor(pro.get("autoridad_demandada")) or _valor(autoridad_demandada)
        _crudo_dem = _valor(f.get("autoridad_demandada")) or dem
        if not unidad:
            unidad, representada = _partir_representacion(_crudo)
            dem = dem or representada
            _crudo_dem = _crudo_dem or representada
        unidad = _con_la_titular(_nombre_en_prosa_rpt(unidad, autoridad=True) or _en_prosa(unidad),
                                 _art_papel)
        dem = _con_la_titular(_nombre_en_prosa_rpt(dem, autoridad=True) or _en_prosa(dem), _crudo_dem) \
            if dem else ""
        if not unidad:
            return "", avisos
        if _RX_TITULAR.match(unidad):
            avisos.append(
                f"«EL TITULAR» VA POR OMISIÓN (promovente = «{unidad}»): el papel no dice el "
                f"artículo del cargo y la legitimación lo escribió en masculino genérico. Si firma "
                f"una mujer, corrígelo («la Titular») en todo el documento.")
        if rep and _mismo_nombre(rep, unidad):
            rep = ""
        # LA FORMA LA DECIDE `tipos_asunto.legitimacion_de` (tercera ronda: la
        # unidad jurídica, la propia demandada «por conducto de» su unidad, o
        # ninguna de las dos), con sus avisos. Si esa pieza aún no sabe de la
        # propia demandada, se resuelve aquí: la unidad es la de su figura, o
        # hueco con aviso; nunca «X, unidad de defensa jurídica de X».
        _av_b: list = []
        if _admite(_ta.legitimacion_de, "avisos"):
            texto = _legitimacion_de(t, unidad, rep, figura=fig, autoridad_demandada=dem,
                                     avisos=_av_b)
            avisos.extend(a_ for a_ in _av_b if a_ not in avisos)
        else:
            if dem and _mismo_nombre(unidad, dem):
                if fig:
                    unidad = fig + (f", {rep}" if rep else "")
                    rep, fig = "", ""
                else:
                    unidad = HUECO
                    avisos.append(
                        "FALTA LA UNIDAD JURÍDICA QUE FIRMÓ EL OFICIO DEL RECURSO (figura_representante): "
                        "la ficha dice que recurre la propia autoridad demandada y el artículo 63 de la "
                        "LFPCA exige que lo haga la unidad encargada de su defensa jurídica. Está en el "
                        "oficio de agravios; la legitimación va en hueco.")
            texto = _legitimacion_de(t, unidad, rep, figura=fig, autoridad_demandada=dem)
        # Al hueco no se le pone artículo: no se sabe qué unidad es.
        texto = re.sub(r"\b(?:el|la)\s+" + re.escape(HUECO), HUECO, texto or "")
        return _con_el_5o_lfpca(texto), avisos
    # AMPARO EN REVISIÓN Y QUEJA
    _parte_cruda = _valor(f.get("promovente"))
    papel = _valor(f.get("caracter")).lower() or (papel_op or "").strip().lower()
    # LA AUTORIDAD QUE RECURRE, COMO ÓRGANO Y CON SU ARTÍCULO (cuarta ronda; AR
    # 208/2025 del banco: «el gobernador del Estado de Querétaro» en la
    # legitimación y «el Gobernador…» en el V I S T O del mismo documento).
    # Por la FORMA del nombre, no por el papel: una persona física que recurre
    # como autoridad no se vuelve órgano («la María…»).
    _es_aut = _ta.es_organo_publico(_sin_rol(_parte_cruda))
    parte = (_en_prosa_con_articulo(_parte_cruda, autoridad=True) if _es_aut
             else _en_prosa(_parte_cruda))
    if not parte:
        return "", avisos
    if rep and _mismo_nombre(rep, parte):
        rep, fig = "", ""
    quejoso = _valor(datos.get("quejoso"))
    recurre_q = None
    if t == "queja":
        if papel:
            recurre_q = papel == "quejoso"
        else:
            try:
                import promovente as _pv_f
                recurre_q = bool(quejoso) and _pv_f.misma_parte(parte, quejoso)
            except Exception:
                recurre_q = bool(quejoso) and _mismo_nombre(parte, quejoso)
        if not _ta.fraccion_5o_de(papel, recurre_q):
            avisos.append(
                "LA FRACCIÓN DEL ARTÍCULO 5o. EN LA LEGITIMACIÓN DE LA QUEJA VA EN HUECO: "
                "no consta en qué carácter recurre " + parte + " —quejosa (fr. I), autoridad "
                "responsable (fr. II), tercera interesada (fr. III) o Ministerio Público "
                "(fr. IV)—. Dilo en el encargo («quién recurre») y vuelve a generar.")
    moral = datos.get("quejoso_moral") if (quejoso and _mismo_nombre(parte, quejoso)) else None
    _recurre_quejosa = papel == "quejoso" or bool(recurre_q) or (
        not papel and bool(quejoso) and _mismo_nombre(parte, quejoso))
    # LA NORMA IMPUGNADA, para la hipótesis del 87 de quien la promulgó (E12,
    # la decide `tipos_asunto.legitimacion_de(..., hay_norma=)`): la razón de
    # `tipos_asunto.norma_impugnada`, o «».
    _norma = None
    if t == "amparo_revision" and _es_aut:
        try:
            _dem_n = f.get("demanda") if isinstance(f.get("demanda"), dict) else {}
            _norma = _ta.norma_impugnada(
                "", [a_ for a_ in (_dem_n.get("actos") or []) if isinstance(a_, str)],
                [a_ for a_ in (list(pro.get("autoridades") or []) or list(_dem_n.get("autoridades") or []))
                 if isinstance(a_, str)]) or ""
        except Exception:
            _norma = None
    _av_b: list = []
    texto = _legitimacion_de(t, parte, rep, figura=fig, moral=moral, papel=papel,
                             recurre_el_quejoso=recurre_q,
                             plural=(_pl_rec if _pl_rec is not None else (_pl_q if _recurre_quejosa else None)),
                             genero=(_g_r or (_g_q if _recurre_quejosa else "")) or None,
                             hay_norma=(bool(_norma) if _norma is not None else None), avisos=_av_b)
    avisos.extend(a_ for a_ in _av_b if a_ not in avisos)
    return texto, avisos


def _organo_descartado(pro: dict) -> bool:
    """¿El compositor descartó el órgano de lo recurrido en el AR? (E10)."""
    pro = pro if isinstance(pro, dict) else {}
    for k in ("organo_recurrido", "juzgado"):
        if str(pro.get(k) or "").strip() == HUECO:
            return True
    return any(_valor(pro.get(k)) or pro.get(k) is True
               for k in ("juzgado_descartado", "organo_descartado"))


def _con_lo_procesal(datos: dict, pro: dict, tipo: str) -> dict:
    """Una copia de `datos` con los nombres de la ficha en su sitio.

    LA RESPONSABLE SE ESCRIBE UNA VEZ. La carátula, la competencia, la
    existencia y el resolutivo leen `datos["responsable"]` (o el órgano
    recurrido); si la ficha la trae, es ésa en todas partes —la verja acusa la
    responsable escrita de dos formas—. En la revisión el órgano es el juzgado
    (`organo_recurrido`); `responsable` sigue siendo la del acto reclamado."""
    d = dict(datos or {})
    t = _ta.normalizar(tipo)
    if t == "amparo_directo":
        if _valor(pro.get("responsable")):
            d["responsable"] = _valor(pro.get("responsable"))
    elif t == "amparo_revision":
        _org = _valor(pro.get("juzgado")) or _valor(pro.get("organo_acto"))
        if _org:
            d["organo_recurrido"] = _org
        # EL ÓRGANO QUE EL COMPOSITOR DESCARTÓ NO VUELVE POR OTRA PUERTA
        # (3-oct-2026, cuarta ronda, E10; AR 307/2024, 448 y 60/2025 recompuestos
        # con las fichas viejas). Si lo recurrido decía que lo dictó una autoridad
        # responsable, el compositor lo descarta y deja hueco en el V I S T O y el
        # trámite; pero `datos["organo_recurrido"]` traía ese mismo nombre y la
        # competencia y la existencia lo escribían. Con la marca del compositor
        # (`organo_recurrido`/`juzgado` = HUECO o `juzgado_descartado`), hueco
        # también en ellas.
        if _organo_descartado(pro):
            d["organo_recurrido"] = HUECO
    else:
        _org = (_valor(pro.get("sala")) if t == "revision_fiscal" else _valor(pro.get("juzgado"))) \
            or _valor(pro.get("organo_acto")) or _valor(pro.get("responsable"))
        if _org:
            d["responsable"] = _org
            if _valor(d.get("organo_recurrido")):
                d["organo_recurrido"] = _org
        elif not _ta.responsable_es_el_organo(t, str(d.get("responsable") or ""),
                                              _valor(pro.get("fraccion_97"))):
            # SIN ÓRGANO EN LA FICHA, LA RESPONSABLE DEL FORMULARIO NO OCUPA SU
            # SITIO (integración, 3-oct-2026, consecuencia de C3 y C4): en la
            # queja y la revisión fiscal la pantalla manda ahí la ordenadora
            # leída del auto de admisión, y la competencia la nombraba como
            # quien dictó lo recurrido mientras el V I S T O lo dejaba en hueco.
            # Hueco también aquí (el compositor ya avisó del dato que falta).
            d["responsable"] = ""
    for k in ("materia", "descripcion_acto", "fecha_acto"):
        if _valor(pro.get(k)):
            d[k] = _valor(pro.get(k))
    # VARIOS QUEJOSOS EN UN CAMPO (3-oct-2026, cuarta ronda, E2; AD 552/2024 del
    # banco: el resultando decía «promovieron» y la carátula «QUEJOSA:» y la
    # legitimación «quien está legitimada»). Lo decide el compositor con UNA
    # regla (`tipos_asunto.es_plural_de_partes`) y la carátula
    # (`tipos_asunto.filas_caratula`) lo lee de lo que él expuso.
    # EN LA REVISIÓN Y LA QUEJA SON DOS NÚMEROS Y NO SE PISAN (3-oct-2026,
    # quinta ronda; AR 208/2025 del banco, regresión de la cuarta): dos quejosos
    # y un solo Gobernador que recurre. Aquí se escribía `datos["plural"]` con el
    # de la QUEJOSA, y la carátula lee `plural` como el de QUIEN RECURRE: salió
    # «AUTORIDADES RESPONSABLES Y RECURRENTE: GOBERNADOR DEL ESTADO DE
    # QUERÉTARO.». Cada número en su clave: el de la quejosa en
    # `plural_quejoso`, el de quien recurre en `plural`. En el amparo directo,
    # como antes: ahí `plural` es el de quien promueve, que es la quejosa.
    if t in ("amparo_revision", "queja"):
        for _k_pl in ("plural_quejoso", "plural"):
            if isinstance(pro.get(_k_pl), bool):
                d[_k_pl] = pro.get(_k_pl)
    else:
        _pl_c = pro.get("plural_quejoso") if isinstance(pro.get("plural_quejoso"), bool) else (
            pro.get("plural") if t == "amparo_directo" else None)
        if isinstance(_pl_c, bool):
            d["plural"] = _pl_c
    if t == "revision_fiscal":
        if _valor(pro.get("fecha_acto")):
            d["fecha_origen"] = _valor(pro.get("fecha_acto"))
        if _valor(pro.get("expediente_tfja")):
            d["expediente_origen"] = _valor(pro.get("expediente_tfja"))
    return d


# CÓMO SE LLAMA CADA MARCADOR EN EL AVISO. Un hueco dentro de un considerando
# se avisa nombrando el DATO, no el marcador: «falta el número del toca», no
# «falta {toca}». `_huecos_bk` los juntaba desde hace semanas y nadie los leía:
# un ********* en la competencia sólo se avisaba en los casos previstos a mano.
_NOMBRE_DEL_DATO = {
    "expediente": "el número del expediente o del juicio de origen",
    "toca": "el número del toca",
    "fecha_acto": "la fecha del acto recurrido",
    "juzgado": "el órgano que dictó lo reclamado o recurrido",
    "juez_distrito": "el juzgado de distrito que dictó la sentencia recurrida",
    "responsable": "la autoridad responsable",
    "fraccion_acuerdo": ("la fracción del punto tercero del Acuerdo General 3/2013 "
                         "(el circuito del tribunal no se pudo leer de su nombre)"),
    "materia": "la materia",
    "inciso": "el inciso del precepto que funda la competencia o la procedencia",
    "descripcion_acto": "qué acto se recurre",
    "supletorio_documentales": "el código supletorio de la Ley de Amparo",
    "tribunal": "el nombre del tribunal",
    "recurrente": "quién recurre",
    "fraccion_63": "la fracción del artículo 63 de la LFPCA",
    "motivo_procedencia": "por qué procede la revisión fiscal",
    "cola_97": "qué resolvió el auto recurrido",
    "objeto": "qué se reclama",
    "concordancia": "la concordancia del órgano",
    # F3 (quinta ronda): la fracción no consta y con ella tampoco la vía.
    # Con su clave entre paréntesis: la verja reconoce así que su hueco genérico
    # del mismo dato ya está avisado (`_hueco_ya_avisado`).
    "fraccion_97": ("la fracción del artículo 97 de la Ley de Amparo y la vía del juicio "
                    "(fraccion_97): I e indirecto si el auto lo dictó un Juzgado de Distrito; "
                    "II y directo si lo dictó la autoridad responsable en un amparo directo"),
}

_RX_ORDINAL_AL_FRENTE = re.compile(
    r"^\s*(?:PRIMERO|SEGUNDO|TERCERO|CUARTO|QUINTO|SEXTO|S[ÉE]PTIMO|OCTAVO|"
    r"NOVENO|D[ÉE]CIMO|[ÚU]NICO)\s*\.\s*")


# ═══════════════════════════════════════════════════════════════════════════
# UN DATO QUE FALTA, UN AVISO (3-oct-2026, tercera ronda)
# ═══════════════════════════════════════════════════════════════════════════
# El banco de oráculo contó hasta cinco avisos por un solo dato (Q 335/2025:
# ocho avisos para dos datos): el genérico de la verja —«HUECO EN EL V I S T O:
# falta…»—, el del compositor que nombra el campo —«FALTA EL ÓRGANO…
# (acto.organo): está en…»—, el «LLEVA HUECO» de cada considerando y el de
# `fase0_oportunidad` para el surtimiento. El útil es el que dice QUÉ campo de
# la ficha falta y DÓNDE está en el papel; los genéricos lo entierran. Se queda
# el específico; el genérico sólo si nadie más avisó de ese hueco, y los que
# quedan se juntan por dato: «falta el órgano: va en hueco en el V I S T O, la
# Interposición y la Competencia».
_CLAVES_DEL_MARCADOR = {
    "expediente": ("acto.expediente", "juicio_amparo", "expediente_tfja", "expediente"),
    "toca": ("acto.toca",),
    "fecha_acto": ("acto.fecha",),
    "juzgado": ("acto.organo", "responsable", "sala"),
    "juez_distrito": ("acto.organo",),
    "responsable": ("responsable", "acto.organo"),
    "recurrente": ("promovente", "caracter"),
    "materia": ("materia",),
    "descripcion_acto": ("acto.sentido",),
    "cola_97": ("acto.sentido",),
    "inciso": ("inciso_97",),
    "fraccion_63": ("fraccion_63",),
    "fraccion_97": ("fraccion_97",),
}


def _sin_tildes_mayus(x) -> str:
    v = unicodedata.normalize("NFD", str(x or "").upper())
    return " ".join("".join(c for c in v if unicodedata.category(c) != "Mn").split())


def _clave_ya_avisada(claves, avisos) -> bool:
    """¿Algún aviso nombra ya ese campo de la ficha, «(acto.fecha)» o
    «(acto.fecha = «…»)»?"""
    for a in avisos or []:
        a = str(a)
        for c in claves or ():
            if c and (f"({c})" in a or f"({c} " in a):
                return True
    return False


_RX_HUECO_VERJA = re.compile(r"^HUECO EN (?P<donde>.+?): falta (?P<dato>.+?) — «")
_RX_VA_EN_HUECO = re.compile(r"[Vv]a en hueco en «([^»]+)»")
_RX_Y_MAS = re.compile(r"\(y (\d+) más en el mismo apartado\)")


def _hueco_ya_avisado(detalle: dict, especificos: list) -> bool:
    """¿El aviso genérico de hueco de la verja (regla «a») repite uno
    específico? Por el campo, si la verja lo da; si no, por el apartado."""
    aviso = str(detalle.get("aviso") or "")
    clave = next((str(detalle.get(k) or "") for k in ("campo", "clave", "dato")
                  if re.fullmatch(r"[a-z_][a-z0-9_.]*", str(detalle.get(k) or ""))), "")
    if clave and _clave_ya_avisada((clave,), especificos):
        return True
    # Varios campos juntos en un aviso (la verja agrupa los de un apartado): cae
    # sólo si TODOS ya están avisados.
    claves = [str(c) for c in (detalle.get("claves") or []) if str(c or "").strip()]
    if len(claves) > 1 and all(_clave_ya_avisada((c,), especificos) for c in claves):
        return True
    m = _RX_HUECO_VERJA.match(aviso)
    if not m:
        return False
    donde = _sin_tildes_mayus(m.group("donde"))
    # UN AVISO QUE NOMBRA VARIOS APARTADOS no se calla por uno de ellos: el
    # específico puede cubrir uno y no los demás.
    if len(re.findall(r"«", donde)) > 1 or len(detalle.get("apartados") or []) > 1:
        return False
    dato = _sin_tildes_mayus(m.group("dato"))
    esp = [_sin_tildes_mayus(a) for a in (especificos or [])]
    if "SURTIMIENTO" in dato and any("SURTIMIENTO" in a for a in esp):
        return True
    if donde.startswith("LOS RESOLUTIVOS"):
        return any("RESOLUTIVO" in a and ("HUECO" in a or "NO IDENTIFICA" in a or "COMODIN" in a)
                   for a in esp)
    m_r = re.search(r"«(.+?)»", donde)
    rot = m_r.group(1).strip() if m_r else re.sub(r"^(?:EL|LA|LOS|LAS)\s+", "", donde)
    if "CONSIDERANDO" in donde:
        w = (re.findall(r"[A-Z]+", rot) or [""])[0]
        return bool(w) and any(w in a and "HUECO" in a for a in esp)
    # V I S T O y resultandos: el compositor dice «Va en hueco en «{apartado}»».
    m_mas = _RX_Y_MAS.search(aviso)
    hace_falta = 1 + (int(m_mas.group(1)) if m_mas else 0)
    n = 0
    for a in especificos or []:
        for ap in _RX_VA_EN_HUECO.findall(str(a)):
            apn = _sin_tildes_mayus(ap)
            if apn and (rot.startswith(apn[:30]) or apn.startswith(rot[:30])):
                n += 1
    return n >= hace_falta


def _fundir_avisos_de_hueco(detalle: list, especificos: list) -> list:
    """Los avisos de la verja, sin los de hueco que repiten uno específico y
    con los que quedan del mismo dato juntos en uno. En el orden de la verja."""
    fuera, grupos = [], {}
    for d in detalle or []:
        aviso = str((d or {}).get("aviso") or "").strip()
        if not aviso:
            continue
        if str(d.get("regla") or "") == "a":
            if _hueco_ya_avisado(d, especificos):
                continue
            m = _RX_HUECO_VERJA.match(aviso)
            if m:
                k = _sin_tildes_mayus(m.group("dato"))
                if k in grupos:
                    i, dondes = grupos[k]
                    dondes.append(m.group("donde"))
                    continue
                grupos[k] = (len(fuera), [m.group("donde")])
        if aviso not in fuera:
            fuera.append(aviso)
    for k, (i, dondes) in grupos.items():
        if len(dondes) > 1:
            junto = ", ".join(dondes[:-1]) + " Y " + dondes[-1]
            fuera[i] = fuera[i].replace(f"HUECO EN {dondes[0]}:", f"HUECO EN {junto}:", 1)
    return fuera


# ═══ UN HECHO, UN AVISO, VENGA DE DONDE VENGA (3-oct-2026, cuarta ronda, E11) ═
# `ficha_tramite.validar`, el compositor y la verja miran la misma ficha y cada
# uno avisaba a su manera: «FECHA IMPOSIBLE: el recurso se presentó el
# 13/05/2025 y la resolución recurrida es del 15/07/2025» y «FECHA IMPOSIBLE: lo
# reclamado/recurrido es de quince de julio… y el escrito se presentó el trece
# de mayo…» (AR 448/2025 del banco), o la fecha de la recurrida que «ESTÁ
# ESCRITA» y que «APARECE» en el acto reclamado. Al final de la composición se
# quedan uno por hecho. EL HECHO SE RECONOCE POR SU DATO —las fechas que nombra,
# el campo de la ficha entre paréntesis—, no por la redacción. Entre dos
# iguales se queda el que vino de fuera (validar, el compositor): el de la
# composición se vuelve a calcular en cada pasada y el de fuera, no.
_RX_CLAVE_ENTRE_PARENTESIS = re.compile(r"\(([a-z_][a-z0-9_]*(?:\.[a-z0-9_]+)*)(?:\)|\s*=)")


def _isos_del_aviso(a: str) -> frozenset:
    try:
        import ficha_tramite as _ft_h
        return frozenset(x[0] for x in _ft_h.fechas_del_texto(a) if x and x[0])
    except Exception:
        return frozenset()


def _clave_del_hecho(aviso) -> tuple:
    """La clave del hecho que dice un aviso, o () si no se reconoce."""
    a = str(aviso or "")
    au = _sin_tildes_mayus(a)
    if au.startswith("FECHA IMPOSIBLE"):
        isos = _isos_del_aviso(a)
        return ("fecha_imposible", isos) if len(isos) >= 2 else ()
    if "FECHA DE LA RESOLUCION RECURRIDA" in au and "EN EL ACTO RECLAMADO" in au:
        isos = _isos_del_aviso(a)
        return ("fecha_en_el_acto", isos) if isos else ()
    if "TITULAR" in au and "POR OMISION" in au:
        # Uno por CAMPO: el del promovente y el de la figura son dos cargos.
        m = re.search(r"\(([a-z_][a-z0-9_.]*)\s*=", a)
        return ("titular_por_omision", m.group(1) if m else "promovente")
    if "PONENTE" in au and "VA POR OMISION" in au:
        return ("ponente_por_omision",)
    # EL ADHESIVO SIN AUTO (E5): el compositor pregunta «¿HUBO AMPARO
    # ADHESIVO?» y no se escriben su considerando ni su punto; la verja, que
    # pide los dos cuando la ficha trae al adherente, decía lo contrario.
    if "ADHESIV" in au and ("¿HUBO" in au or "CONSTA Y NO SE TRATA" in au):
        return ("hubo_adhesivo",)
    if au.startswith("DATO TRUNCADO"):
        m = re.search(r"\(([a-z_][a-z0-9_.]*)\s*=", a)
        if m:
            return ("truncado", m.group(1))
    if au.startswith("FALTA "):
        m = _RX_CLAVE_ENTRE_PARENTESIS.search(a)
        if m:
            return ("falta", m.group(1))
    return ()


def _un_aviso_por_hecho(avisos: list, de_composicion=()) -> list:
    """Los avisos, sin los que repiten un hecho ya dicho (`_clave_del_hecho`).
    Conserva el orden; entre dos del mismo hecho se queda la pregunta
    «¿HUBO…?» (la que dice qué hacer), después el de fuera de
    `de_composicion` y, a igualdad, el primero."""
    comp = set(de_composicion or ())
    ganador: dict = {}
    for i, a in enumerate(avisos or []):
        k = _clave_del_hecho(a)
        if not k:
            continue
        rango = (0 if "¿HUBO" in str(a) else 1, 1 if a in comp else 0, i)
        if k not in ganador or rango < ganador[k][0]:
            ganador[k] = (rango, a)
    fuera = []
    for a in avisos or []:
        if a in fuera:
            continue
        k = _clave_del_hecho(a)
        if k and ganador[k][1] != a:
            continue
        fuera.append(a)
    return fuera


def apoyo_dispensa(tipo: str) -> str:
    """La cita de la 2a./J. 58/2010 para la dispensa de este tipo.

    «POR ANALOGÍA» SÓLO EN LA REVISIÓN FISCAL (3-oct-2026). La jurisprudencia
    habla de las sentencias de AMPARO: en el amparo directo, la revisión y la
    queja se aplica directamente, y decir «por analogía» es decir que no es su
    materia. Sólo en la revisión fiscal —que no es amparo— la analogía es
    exacta (oro_RF: 12 de 59 lo dicen así)."""
    if _ta.normalizar(tipo) == "revision_fiscal":
        return APOYO_DISPENSA
    return APOYO_DISPENSA.replace(", por analogía,", "")


_CLASE_RESOLUTIVO_AD = {"laudo": ("el laudo", "dictado"),
                        "resolucion": ("la resolución", "dictada"),
                        "sentencia": ("la sentencia", "dictada")}


def _cola_resolutivo_ad(pro: dict, ficha: dict, avisos: list) -> str:
    """«contra la sentencia dictada el {fecha}, por {responsable}, en el toca
    {t}, derivado del expediente {e}» — la cola del resolutivo del amparo
    directo con los datos de la ficha (`datos["procesal"]`), los mismos que el
    V I S T O. Lo que falte va en hueco y se avisa una vez nombrando el dato."""
    acto = (ficha or {}).get("acto") if isinstance((ficha or {}).get("acto"), dict) else {}
    clase = _valor(acto.get("clase")).lower()
    clase = clase.replace("ó", "o")
    if not clase:
        _desc = _valor(pro.get("descripcion_acto")).lower()
        clase = ("laudo" if "laudo" in _desc else "resolucion" if "resoluci" in _desc
                 else "sentencia")
    art, dictad = _CLASE_RESOLUTIVO_AD.get(clase, _CLASE_RESOLUTIVO_AD["sentencia"])
    faltan = []
    fecha = _valor(pro.get("fecha_acto"))
    if not fecha:
        faltan.append("la fecha del acto reclamado")
    resp = _con_articulo(_valor(pro.get("responsable")) or _valor(pro.get("organo_acto")))
    if not resp:
        faltan.append("la autoridad responsable")
    toca_p = _valor(pro.get("toca_en_prosa")) or (
        f"toca {_valor(pro.get('toca'))}" if _valor(pro.get("toca")) else "")
    exp_p = _valor(pro.get("expediente_en_prosa")) or (
        f"expediente {_valor(pro.get('expediente'))}" if _valor(pro.get("expediente")) else "")
    if toca_p and exp_p:
        donde = f"{toca_p}, derivado del {exp_p}"
    else:
        donde = toca_p or exp_p
    if not donde:
        faltan.append("el toca o el expediente en que se dictó")
    if faltan:
        avisos.append(
            "EL RESOLUTIVO DEL AMPARO DIRECTO NO IDENTIFICA EL ACTO POR COMPLETO: falta "
            + " y ".join(faltan) + ". Va en hueco; está en la sentencia reclamada y en "
            "«Trámite en este tribunal» (fecha y órgano del acto, toca o expediente).")
    return _contraer(f"contra {art} {dictad} el {fecha or HUECO}, por {resp or HUECO}, "
                     f"en el {donde or HUECO}")


def _puntos_con_firmeza(puntos: list) -> list:
    """«Queda firme el sobreseimiento…» delante, «En la materia de la revisión,
    se confirma…» y fuera el «Se sobresee» que repetía lo que ya está firme.
    Renumerados (AR, 3-oct-2026; oro_AR: «Queda firme + En la materia de la
    revisión se confirma + ampara», 72 resolutivos de 2025-26)."""
    cuerpos = [_RX_ORDINAL_AL_FRENTE.sub("", p) for p in (puntos or [])]
    fuera = [_ta.FIRME_SOBRESEIMIENTO]
    for c in cuerpos:
        if re.match(r"\s*Se\s+sobresee\b", c, re.I):
            continue
        if re.match(r"\s*Se\s+confirma\s+la\s+sentencia\s+recurrida\.?\s*$", c, re.I):
            c = "En la materia de la revisión, se confirma la sentencia recurrida."
        fuera.append(c)
    return [f"{o}. {c}" for o, c in zip(_ORDINALES, fuera)]


def componer(datos: dict, estructura: Estructura, computo, fecha_en_letra,
             ruta_salida: str, antecedentes=None, resumen_acto=None,
             resumen_conceptos=None, problemas=None, estudio=None,
             calificaciones=None, tesis=None, marco_escrito="",
             tipo_asunto="amparo_directo", normas=None, criterios=None,
             sintesis=None) -> str:
    """Escribe el .docx entero. No hay plantilla de la que partir."""
    # LA FICHA DE TRÁMITE MANDA (3-oct-2026, bandera `procedencia_por_tipo`):
    # `_pro` es lo que el compositor de resultandos sacó de ella, `_ficha_t`
    # la ficha entera. Vacíos sin la bandera: el camino de siempre.
    _rige_pt = _rige_procedencia()
    _pro = (dict(datos.get("procesal") or {})
            if _rige_pt and isinstance(datos.get("procesal"), dict) else {})
    if not any(_valor(v) or isinstance(v, dict) for v in _pro.values()):
        _pro = {}
    _ficha_t = (dict(datos.get("tramite") or {})
                if _rige_pt and isinstance(datos.get("tramite"), dict) else {})
    if _pro:
        datos = _con_lo_procesal(datos, _pro, tipo_asunto)
    _tn = _ta.normalizar(tipo_asunto)
    # EL CAMINO NUEVO ES BANDERA Y FICHA (3-oct-2026, tercera ronda): con la
    # bandera y sin ficha el documento es el de siempre.
    _nuevo = bool(_pro or _ficha_t)
    # EL NOMBRE DE LA PARTE EN LOS RESOLUTIVOS, COMO EN LA PROSA: sin las
    # versales ni la etiqueta de rol de la carátula (AD 274/2025 de punta a
    # punta: «no ampara ni protege a GABRIEL REYES ALAMO» mientras el V I S T O
    # lo nombraba en mayúsculas y minúsculas).
    # Y CON SU ARTÍCULO SI ES COLECTIVO (3-oct-2026, cuarta ronda, E1; AD
    # 335/2025 y AR 60/2025: «no ampara ni protege a Sucesión a Bienes de…»
    # mientras el V I S T O decía «la Sucesión»): la misma puerta que el
    # compositor, `resultandos_por_tipo.nombre_en_prosa`.
    _q_prosa = (_en_prosa_con_articulo(datos.get("quejoso")) if _nuevo
                else str(datos.get("quejoso") or ""))
    _avisos_caratula: list = []
    _ponente_c = None
    if _nuevo:
        datos = dict(datos)
        # LA CARÁTULA SIN EL NÚMERO DEL ASUNTO (AR, Q y RF del banco): con el
        # encabezado vacío no salía «AMPARO EN REVISIÓN …: N/AAAA» ni había
        # aviso; la ficha trae el tipo, la materia y el número para componerlo,
        # igual que lo compone el formulario (`tipos_asunto.encabezado_de`).
        if not _valor(datos.get("encabezado")):
            _num_e = _valor(_ficha_t.get("numero")) or _valor(datos.get("numero"))
            _mat_e = (_valor(_ficha_t.get("materia")) or _valor(_pro.get("materia"))
                      or _valor(datos.get("materia")))
            if _num_e:
                datos["encabezado"] = re.sub(r"\s+:", ":", " ".join(
                    _ta.encabezado_de(tipo_asunto, _mat_e, _num_e).split()))
            else:
                _avisos_caratula.append(
                    "LA CARÁTULA NO IDENTIFICA EL ASUNTO: no hay encabezado ni número en la "
                    "ficha de trámite. Escribe el número del asunto en el formulario.")
        # EL ADHERENTE EN LA CARÁTULA (AD 279 y RF 4/2025 del banco): la ficha
        # lo trae y el resultando, el considerando y el punto lo usan, pero la
        # carátula no tenía de dónde tomarlo.
        # SIN NINGÚN AUTO DEL ADHESIVO NO SE AFIRMA UN ADHERENTE (3-oct-2026,
        # integración, E5): AD 274/2025 del banco, la tercera interesada sólo
        # presentó alegatos; el resultando pregunta «¿HUBO…?» y la carátula no
        # debe darlo por hecho.
        if _tn in ("amparo_directo", "amparo_revision", "revision_fiscal") \
                and not _valor(datos.get("adherente")) and not _pro.get("adhesivo_sin_auto"):
            _adh_c = _pro.get("adherente")
            if isinstance(_adh_c, dict):
                _adh_c = _adh_c.get("quien")
            if not _valor(_adh_c) and isinstance(_ficha_t.get("adhesivo"), dict):
                _adh_c = _ficha_t["adhesivo"].get("quien")
            if _valor(_adh_c):
                datos["adherente"] = _en_prosa(_adh_c)
        _ponente_c = _ponente_de_caratula(datos, _ficha_t, _pro)
        _avisos_caratula.extend(_ponente_c[2])
    doc = docx.Document()
    notas: list = []
    _pagina(doc)
    _encabezado(doc, datos.get("encabezado", ""))
    # Los avisos deterministas del documento, en un solo sitio y declarados
    # ANTES del primero que los usa: la lista se llenaba en dos puntos y se
    # declaraba entre ellos, que en Python es un UnboundLocalError esperando.
    _avisos_bk: list = []
    # LOS DE LA COMPOSICIÓN ANTERIOR SE RETIRAN (28-sep-2026, AR 631/2025): la
    # estructura se reutiliza del adelanto y sus avisos de rama se quedaban.
    try:
        _previos = set(getattr(estructura, "avisos_de_composicion", None) or [])
        if _previos:
            estructura.avisos = [a for a in (estructura.avisos or []) if a not in _previos]
        estructura.avisos_de_composicion = []
    except Exception:
        pass
    # LOS AVISOS QUE TRAE LA ESTRUCTURA SON DE OTRAS PIEZAS (el compositor de
    # resultandos nombra el campo de la ficha que falta): sirven para no
    # repetirlos con un aviso genérico (ver `_fundir_avisos_de_hueco`).
    _avisos_de_fuera = [str(a_) for a_ in (getattr(estructura, "avisos", None) or [])]

    # ═══════════════════════════════════════════════════════════════════════
    # UN SOLO EMBUDO PARA TODO LO QUE ESCRIBIÓ EL MODELO
    # ═══════════════════════════════════════════════════════════════════════
    # El meta-lenguaje se limpiaba en las fases de lectura, que es donde salió,
    # pero al documento llegan CINCO salidas de modelo por caminos distintos:
    # los resúmenes, la estructura, el estudio de fondo, el marco jurídico y la
    # propuesta. Parchear cada fase deja siempre una puerta abierta —ya van
    # tres rondas encontrando la siguiente—, así que se filtra aquí, que es por
    # donde pasa todo lo que acaba en el .docx, sea cual sea su origen.
    try:
        import meta_lenguaje as _mlc

        def _limpio(x):
            t, q = _mlc.limpiar(x or "")
            for f in q:
                _avisos_bk.append(
                    f"SE QUITÓ UNA FRASE QUE HABLABA DEL ARCHIVO, NO DEL "
                    f"ASUNTO: «{f[:200]}». Va entera aquí por si el filtro se "
                    f"equivocó y hay que devolverla.")
            return t

        estudio = _limpio(estudio) if isinstance(estudio, str) else estudio
        if isinstance(estudio, str):
            estudio = _lenguaje_de_proyecto(estudio)
        elif isinstance(estudio, (list, tuple)):
            estudio = [_lenguaje_de_proyecto(str(x)) for x in estudio]
        marco_escrito = _limpio(marco_escrito)
        if estructura is not None:
            estructura.apertura = _limpio(getattr(estructura, "apertura", ""))
            estructura.visto = _limpio(getattr(estructura, "visto", ""))
            estructura.competencia = _limpio(getattr(estructura, "competencia", ""))
            estructura.existencia = _limpio(getattr(estructura, "existencia", ""))
            estructura.procedencia = _limpio(getattr(estructura, "procedencia", ""))
            for _r in (estructura.resultandos or []):
                if isinstance(_r, dict):
                    _r["texto"] = _limpio(_r.get("texto", ""))
        if isinstance(antecedentes, list):
            antecedentes = [_limpio(x) for x in antecedentes]
        elif isinstance(antecedentes, str):
            antecedentes = _limpio(antecedentes)
    except Exception:
        pass

    # LA SUPLENCIA POR UNA MATERIA QUE NO ES LA DEL ASUNTO. Ver la nota en
    # tipos_asunto: no es un defecto de la máquina —la parte lo invocó— pero el
    # proyecto no lo decía, y quien lee no puede distinguir el disparate de la
    # parte del de la máquina sin ir al escrito. Además, esa petición hay que
    # contestarla.
    try:
        _sup = _ta.suplencia_de_otra_materia(
            " ".join(str(x) for x in [resumen_conceptos, estudio] if x),
            str(datos.get("materia") or ""))
        for _m in _sup:
            _avisos_bk.append(
                f"LA PARTE PIDE SUPLENCIA POR MATERIA {_m.upper()} y este "
                f"asunto no lo es. Está en SU escrito, no lo puso el sistema; "
                f"pero es una petición que hay que contestar, normalmente "
                f"declarándola improcedente, y el proyecto no la contesta.")
    except Exception:
        pass

    # LA VERJA EMPIEZA LIMPIA EN CADA PROYECTO. Es una lista de módulo —como
    # `avisos_ensamblado`— y con dos workers un documento heredaría los avisos
    # del anterior.
    avisos_cotejo.clear()
    # LOS ASUNTOS RELACIONADOS QUE MARCÓ EL SECRETARIO (C6, 3-oct-2026): el
    # renglón del rubro aquí y el considerando antes de la dispensa. Nunca
    # conexidad en automático: sin lista, ni lo uno ni lo otro.
    _rel_lista, _rel_mat, _rel_num = (_relacionados_del_asunto(tipo_asunto, datos, _pro, _ficha_t)
                                      if _rige_pt else ([], "", ""))
    # Y EN EL ENCABEZADO DE CADA PÁGINA (3-oct-2026). Los tres engroses del banco
    # con relacionados, de tres ponencias distintas (AD 456/2025, AD 552/2024,
    # Q 24/2026), repiten «RELACIONADO CON…» junto al número en cada hoja: es
    # práctica del tribunal, y sin ella el secretario lo añadiría a mano (David:
    # «que no sea necesario adecuar nada»). Se reescribe el encabezado ya puesto.
    if _rel_lista and str(datos.get("encabezado") or "").strip():
        _encabezado(doc, " ".join([
            str(datos.get("encabezado") or "").strip().rstrip("."),
            _ta.rotulo_relacionados(_rel_lista, _rel_mat).strip().rstrip(".")]))
    avisos_doc = _caratula(doc, datos, tipo_asunto,
                           ponente=(_ponente_c[:2] if _ponente_c else None), nuevo=_nuevo,
                           relacionados=(_ta.rotulo_relacionados(_rel_lista, _rel_mat)
                                         if _rel_lista else ""))
    avisos_doc = list(avisos_doc) + _avisos_caratula

    # LA FÓRMULA MANDA SOBRE LO QUE ESCRIBA EL MODELO. Si tenemos ciudad y
    # tribunal —y los tenemos siempre, son del encargo— el proemio se compone.
    _ap = _apertura_compuesta(datos) or estructura.apertura
    if _ap:
        parrafo(doc, _ap, sangria=True)
    if estructura.visto:
        # El rótulo lo pone la composición; el modelo lo repite igual aunque se
        # le pida que no —«V I S T O, VISTO, para resolver…»—. Se le quita.
        _v = re.sub(r"^\s*V\s*I\s*S\s*T\s*O\s*S?\s*,?\s*", "",
                    estructura.visto.strip(), flags=re.I)
        # «V I S T O S» en la revisión fiscal: 28 de 28 en el corpus. Se
        # imprimía en singular a máquina y encima se limpiaba con una regex que
        # admite la S, así que aunque el modelo acertara se le borraba: este
        # tipo era incorregible por prompt.
        # «V I S T O, Para resolver…»: el modelo escribe su trozo como si
        # empezara una frase, porque para él lo es. Detrás de la coma del
        # rótulo va minúscula, y el corpus lo escribe así en los cuatro.
        if _v[:1].isupper() and _v[:3] not in ("V I",):
            _v = _v[:1].lower() + _v[1:]
        tramos(doc, [(_ta.proemio_de(tipo_asunto)["rotulo"], {"bold": True}),
                     (_v, {})], sangria=True)

    # ═══ LA ESTRUCTURA, MEDIDA SOBRE 58 ENGROSES ═══════════════════════
    # 26 amparos directos civiles, 16 administrativos y 16 revisiones. En
    # NINGUNO existe un considerando llamado «Consideraciones de la sentencia
    # reclamada», ni «Planteamientos de la parte quejosa», ni «Marco jurídico»
    # —éste aparece 1 vez en 58 y como SUBTÍTULO dentro del Estudio—. Todo eso
    # vive DENTRO del considerando de Estudio, en subtítulos en negrita SIN
    # ordinal. El redactor los emitía como considerandos propios y por eso el
    # estudio acababa en OCTAVO, chocando con su propio encabezado.
    #
    # LOS ORDINALES SE CALCULAN, NO SE ESCRIBEN. Se numera al final la lista de
    # apartados realmente emitidos. Así el Estudio cae en QUINTO cuando no hay
    # Antecedentes —que es lo que dijo David— y en SEXTO cuando sí los hay, sin
    # que nadie tenga que decidirlo.
    def _emitir(apartados):
        """Numera y escribe. Cada apartado es (rótulo, escritor)."""
        for k, (rot, escribir_cuerpo) in enumerate(apartados):
            # SEPARACIÓN POR ESPACIADO, NO POR PÁRRAFO VACÍO. Un párrafo en
            # blanco entre cada apartado y cada cita dejaba huecos enormes en
            # el papel —22 de 173 párrafos eran aire— y además se arrastran al
            # editar. Word tiene `space_before` justo para esto.
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(14)
            r1 = p.add_run(f"{_ORDINALES[min(k, 9)]}. ")
            r1.bold = True
            r2 = p.add_run(f"{rot} ")
            r2.bold = True
            # TODO EL PROYECTO CON SANGRÍA, también el párrafo que abre cada
            # apartado: el ordinal en negrita no lo exime. Lo pidió David y es
            # lo que hace el corpus.
            _fmt(p, sangria=True)
            # EL RÓTULO SE ATA A SU CUERPO; UN CUERPO NO SE ATA A LO SIGUIENTE.
            # `keep_with_next` quedaba puesto en este párrafo SIEMPRE, y como
            # el cuerpo del apartado se escribe DENTRO de él, lo que Word leía
            # era «no separes este párrafo largo del que viene detrás»: si el
            # siguiente no cabía, empujaba media página en blanco. Es el hueco
            # que David vio detrás de la tabla del cómputo, y es exactamente el
            # mismo fallo que ya se corrigió en el bloque de las tesis —donde
            # atar la cadena entera dejaba media hoja vacía antes de cada
            # jurisprudencia— sin que la lección llegara hasta aquí.
            #
            # Se ata sólo mientras el párrafo es un rótulo solo: si el cuerpo
            # entra en él, deja de estarlo. Un párrafo que ya lleva su propio
            # texto no puede quedar huérfano.
            _solo_rotulo = len(p.text)
            p.paragraph_format.keep_with_next = True
            escribir_cuerpo(p)
            if len(p.text) > _solo_rotulo + 40:
                p.paragraph_format.keep_with_next = False

    def _texto_en(p, texto, resto=None):
        """El primer párrafo continúa el rótulo; el resto van aparte.

        Es el embudo: todo lo que el modelo escribe entra al documento por
        aquí, así que el andamio se quita aquí una vez y no en catorce sitios.
        """
        texto = sin_andamio(texto or "")
        # ═══════════════════════════════════════════════════════════════════
        # UN SALTO DE LÍNEA DENTRO DE UN PÁRRAFO JUSTIFICADO ESTIRA EL TEXTO
        # ═══════════════════════════════════════════════════════════════════
        # David: «saltos de renglones para que el texto no se estirara
        # innecesariamente en todo el párrafo en algunas partes (revisa el
        # adelanto que produjo el pipeline y verás)».
        #
        # Y se ve. El modelo escribe el resultando primero con sus sub-bloques
        # separados por `\n` —«AUTORIDAD RESPONSABLE:», la lista, «ACTOS
        # RECLAMADOS:»— y esto lo metía TODO en un párrafo, convirtiendo cada
        # salto en un `<w:br>`. Word, al justificar, reparte la última línea
        # antes de cada salto de margen a margen: «AUTORIDAD RESPONSABLE:»
        # queda con dos palabras separadas por diez centímetros de blanco.
        # Medido: 11 saltos en mi documento, CERO en el suyo.
        #
        # La línea corta no es el problema —es el rótulo de un bloque, y está
        # bien que sea corta—: el problema es pedirle a Word que la justifique.
        # Cada trozo va a su propio párrafo y termina donde termina.
        _trozos = [z.strip() for z in re.split(r"\n+", texto) if z.strip()]
        if _trozos:
            r = p.add_run(_trozos[0])
            r.font.name = FUENTE
            # El tamaño se hereda del estilo Normal, igual que en el resto del
            # documento y que en el suyo: sólo se estampa lo que se aparta.
        for x in _trozos[1:]:
            # EL RÓTULO DE UN SUB-BLOQUE VA EN NEGRITA. En el suyo,
            # «AUTORIDAD RESPONSABLE:» y «ACTOS RECLAMADOS:» pesan y su
            # contenido no: es lo que permite encontrarlos sin leer, que era
            # justo la razón de partirlos en bloques.
            #
            # SE RECONOCE POR LA FORMA, no por una lista de rótulos: una línea
            # corta, en versales y terminada en dos puntos es un rótulo en
            # cualquier tipo de asunto y en cualquier circuito. Una lista de
            # nombres habría que ampliarla cada vez que un secretario rotule
            # algo distinto.
            if _ES_ROTULO_BLOQUE.match(x):
                parrafo(doc, x, negrita=True)
            else:
                parrafo(doc, x)
        for x in (resto or []):
            x = sin_andamio(x)
            if x.strip():
                parrafo(doc, x.strip())

    # ── RESULTANDO ──
    rotulo(doc, "Resultando", dos_puntos=_nuevo)
    res_apartados = []
    # LA PERÍFRASIS EN UN RESULTANDO ES EL APARTADO SIN CUMPLIR. Se avisa —no
    # se borra: la frase ocupa el sitio de algo que debe escribirse, y quitarla
    # dejaría el resultando mudo—. Y arrastra un segundo daño que conviene
    # nombrar en el mismo aviso: el número de expediente del considerando de
    # existencia se lee de estos resultandos.
    try:
        import meta_lenguaje as _ml
        _ev = _ml.perifrasis(" ".join(str(r.get("texto") or "")
                                      for r in (estructura.resultandos or [])))
        # LA COLA DE LA EXISTENCIA, SÓLO DONDE HAY EXISTENCIA (revisión AR,
        # 3-oct-2026): C1 la quitó del amparo en revisión, y el aviso la seguía
        # nombrando en todos los tipos.
        try:
            _con_exi = bool((ESQUELETO.get(_ta.normalizar(tipo_asunto)) or {}).get("existencia", True))
        except Exception:
            _con_exi = True
        for _e in _ev:
            _avisos_bk.append(
                f"UN RESULTANDO EVADE EL DATO: «{_e}». Los resultandos existen "
                f"para individualizar —fecha, órgano, expediente, nombre—"
                + (", y una perífrasis ahí deja además sin número el expediente de "
                   "origen del considerando de existencia." if _con_exi else "."))
    except Exception:
        pass

    for res in (estructura.resultandos or []):
        cuerpo = (res.get("texto") or "").strip()
        if not cuerpo:
            continue
        cuerpo = _con_el_sentido_del_a_quo(
            cuerpo, str(datos.get("resolvio_a_quo") or ""))
        rot = (res.get("titulo") or "").strip().rstrip(".") + "."
        res_apartados.append((rot, (lambda c: lambda p: _texto_en(p, c))(cuerpo)))
    # La sesión SIEMPRE cierra el resultando y enlaza con el considerando.
    # LA FÓRMULA ESTABA MEDIDA EN EL BANCO Y NADIE LA LEÍA. Otra vez: 43 de 44
    # documentos del propio tribunal escriben «El presente asunto se listó el
    # {fecha}, para verse en sesión ordinaria de {fecha} siguiente; lo
    # anterior, de conformidad con el Acuerdo General…». Aquí se escribía una
    # versión corta de mi cosecha —«se listó para la sesión de *********, la
    # cual se celebró conforme a las disposiciones aplicables»— que fundía LAS
    # DOS FECHAS en un solo hueco y se comía la cita del acuerdo, que es lo que
    # da fundamento a que la sesión sea remota.
    #
    # LOS DOS HUECOS SE QUEDAN, y son honestos: el proyecto se redacta ANTES de
    # la sesión, así que ni el secretario sabe todavía esas fechas. Lo que se
    # arregla es que sean DOS huecos con su forma y su fundamento alrededor, y
    # no un agujero que se traga medio párrafo.
    #
    # LA COLA NORMATIVA: el banco mide dos, y la frontera es abril de 2026. La
    # del Acuerdo 6/2026 rige los asuntos listados a partir de entonces (20/44)
    # y es la vigente; la cola COVID (23/44) es la de los anteriores. Como la
    # fecha de sesión no consta, se escribe la vigente y se avisa.
    _rot_sesion = ("Verificación de la sesión vía remota."
                   if _ta.normalizar(tipo_asunto) in ("queja", "amparo_revision")
                   else "Celebración de la sesión vía remota.")
    _cola_sesion = (
        "lo anterior, de conformidad con el Acuerdo General 6/2026, del Pleno "
        "del Órgano de Administración Judicial que regula la integración y "
        "trámite del expediente electrónico y el uso de videoconferencias en "
        "todos los asuntos competencia de los órganos jurisdiccionales a cargo "
        "del Órgano, publicado en el Diario Oficial de la Federación el "
        "diecisiete de abril de dos mil veintiséis; y,")
    res_apartados.append((
        _rot_sesion,
        lambda p: _texto_en(p, f"El presente asunto se listó el "
                               f"{datos.get('fecha_lista') or HUECO}, para "
                               f"verse en sesión ordinaria de "
                               f"{datos.get('fecha_sesion') or HUECO} "
                               f"siguiente; {_cola_sesion}")))
    if not (datos.get("fecha_lista") and datos.get("fecha_sesion")):
        # Y NO SE PIDEN. David las retiró del formulario el mismo día que se
        # añadieron: «no me sirven porque estas, al final, quedarán hasta el
        # momento en que se revisen por los magistrados». El hueco no es un
        # fallo del sistema, es el estado real del asunto cuando se redacta.
        _avisos_bk.append(
            "LAS DOS FECHAS DE LA SESIÓN VAN EN HUECO, y así tienen que ir: "
            "cuándo se listó el asunto y para qué sesión se fijan cuando los "
            "magistrados lo revisan, después de que esto se escriba. Se "
            "rellenan al engrosar.")
    _avisos_bk.append(
        "COMPRUEBA LA COLA NORMATIVA del párrafo de la sesión: se escribió la del "
        "Acuerdo General 6/2026 del Pleno del Órgano de Administración Judicial, "
        "en vigor desde el dieciocho de abril de dos mil veintiséis (abrogó el "
        "12/2020); si la sesión se celebró antes de esa fecha, la cola es la de "
        "los Acuerdos 16/2009 y 12/2020 del otrora Consejo de la Judicatura "
        "Federal.")
    _emitir(res_apartados)

    # ── CONSIDERANDO ──
    rotulo(doc, "Considerando", dos_puntos=_nuevo)
    cs = [str(c or "").strip().lower() for c in (calificaciones or []) if c]
    # «INNECESARIO» NO CONCEDE NI NIEGA. Empieza por «i» como «infundado» e
    # «inoperante», pero significa otra cosa: que no se entra al planteamiento.
    # Lo que decide el sentido del fallo es el PRINCIPAL, y si ése es fundado
    # el asunto se concede aunque los accesorios queden sin materia.
    # POR EL PREDICADO, NO POR LA CADENA. Aquí se decide si el amparo SE
    # CONCEDE, y `startswith("fundad")` devolvía False para «esencialmente
    # fundado» —que es el 23% de los agravios en las revisiones que revocan—.
    # Una concesión compuesta como negativa no falla: se firma.
    concede = any(_ta.prospera(c) for c in cs)
    esq = esqueleto_de(tipo_asunto)
    q = esq["q"]
    con_apartados = []

    # ═══ EL BANCO MANDA DONDE LA REDACCIÓN ES FORMAL ═══════════════════
    # La competencia, la existencia, la legitimación y la dispensa tienen UNA
    # frase en el oficio, con su cadena de fundamentos en un orden que no es
    # casual. El modelo escribía una versión correcta y anodina, y el
    # secretario la reescribía entera: entonces el adelanto no le ahorró nada.
    # Medido sobre 363 documentos. El estudio y los antecedentes NO llevan
    # plantilla: ahí no hay fórmula que valga.
    import banco as _bk
    _datos_bk = dict(datos)
    _datos_bk.setdefault("q", q)
    # LOS MARCADORES QUE SÍ SE DEDUCEN. Salieron en hueco la primera vez y no
    # tenían por qué: la materia está en el encabezado —«AMPARO DIRECTO CIVIL
    # 380/2025»—, la concordancia depende del género de la autoridad, y el
    # inciso lo manda la materia: c) en civil y mercantil, b) en administrativa
    # y agraria. Medido en el corpus. Un hueco que se puede rellenar es trabajo
    # que se le deja al secretario sin motivo.
    _resp = str(datos.get("responsable", ""))
    # EN LA QUEJA Y LA REVISIÓN FISCAL LA RESPONSABLE DEL FORMULARIO NO SIEMPRE
    # ES EL ÓRGANO (integración, 3-oct-2026, consecuencia de C3 y C4;
    # `tipos_asunto.responsable_es_el_organo`): sin el renglón del órgano en la
    # carátula, ese campo trae la ordenadora leída del auto de admisión, y la
    # competencia y la procedencia la nombraban «dictado por» ella. Si la ficha
    # no dio el órgano (`_con_lo_procesal` lo pone en `responsable` cuando sí),
    # manda el que leyó la ficha de partes; sin él, hueco.
    if (_ta.normalizar(tipo_asunto) in ("queja", "revision_fiscal")
            and not any(_valor(_pro.get(k_)) for k_ in ("organo_acto", "juzgado", "sala"))
            and not _ta.responsable_es_el_organo(tipo_asunto, _resp, _valor(_pro.get("fraccion_97")))):
        _resp = str(datos.get("organo_recurrido") or "").strip()
    _mat = (datos.get("materia") or "").strip().lower()
    if not _mat:
        _enc = str(datos.get("encabezado", "")).lower()
        for m_, clave in (("civil", "civil"), ("administrativ", "administrativa"),
                          ("mercantil", "mercantil"), ("laboral", "laboral"),
                          ("familiar", "familiar"), ("agrari", "agraria")):
            if m_ in _enc:
                _mat = clave
                break
    # LA MATERIA NO PUEDE QUEDAR EN HUECO en mitad de la competencia. Si no se
    # dedujo del encabezado ni del tribunal, se dice la del tipo de asunto, que
    # es cierta: una revisión fiscal es administrativa por definición.
    _datos_bk.setdefault("materia", _mat or {
        "revision_fiscal": "administrativa", "queja": "administrativa",
    }.get(str(tipo_asunto or "").strip().lower(), "") or HUECO)
    # LA LETRA DEL INCISO ES LA DE LA MATERIA en el 107, fr. V, constitucional
    # y en el 35, fr. I, de la Ley Orgánica: penal a), administrativa b), civil
    # c), laboral d). Se mandaba todo lo que no fuera administrativo o agrario a
    # la c), y una materia vacía se firmaba como civil (verificación de normas,
    # 27-sep-2026). Sin materia conocida, hueco a la vista.
    _datos_bk.setdefault("inciso", {
        "penal": "a", "administrativa": "b", "agraria": "b", "fiscal": "b",
        "civil": "c", "familiar": "c", "mercantil": "c",
        "laboral": "d", "trabajo": "d"}.get(_mat, HUECO if not _mat else "c"))
    # EL INCISO DEL 97 NO ES EL DE LA MATERIA. Ver la nota en tipos_asunto: un
    # solo marcador servía a dos preceptos que se reparten por cosas distintas,
    # y la queja acababa fundando su procedencia en el supuesto equivocado.
    _fr97 = "I"
    # LA FRACCIÓN DEL 97 NO SE SUPONE (3-oct-2026, quinta ronda, F3; Q 335/2025
    # del banco recompuesto sin la fracción): el compositor dejaba la vía en
    # hueco en el V I S T O y avisaba, y aquí `or "I"` la daba por sabida: la
    # competencia decía «dictado en un juicio de amparo indirecto… 97, fracción
    # I» y la procedencia «en el juicio de amparo indirecto», y el aviso del
    # plazo de dos días empujaba a recontar con dos un recurso que tiene cinco.
    # Con la ficha, la fracción es la que consta (`_fraccion_97_de`); si no
    # consta, «» y la fracción y la vía van en hueco en los dos considerandos.
    # Sin la ficha, como siempre.
    if _ta.normalizar(tipo_asunto) == "queja" and _nuevo:
        _fr97 = _fraccion_97_de(_pro, _ficha_t)
    _fr97_en_hueco = _ta.normalizar(tipo_asunto) == "queja" and _nuevo and not _fr97
    # «, FRACCIÓN I,» en los avisos sólo si consta.
    _fr97_aviso = f", FRACCIÓN {_fr97}," if _fr97 else ""
    # UN HECHO, UN AVISO (E11; Q 335/2025: el del inciso salía dos veces, el
    # del compositor y éste, y éste afirmaba «FRACCIÓN I» sin saberla).
    _inciso_ya_avisado = _clave_ya_avisada(("inciso_97",), _avisos_de_fuera)
    if _ta.normalizar(tipo_asunto) == "queja" and _pro:
        # EL INCISO Y SU COLA SALEN DE LA FICHA, UNA SOLA VEZ, para la
        # competencia y para la procedencia (3-oct-2026). La ficha los toma del
        # auto recurrido y de la vía; la fracción II es la del amparo directo
        # (actos de la responsable) y lleva su propia cadena.
        _i97 = _valor(_pro.get("inciso_97")).lower().rstrip(")")
        _datos_bk["inciso"] = _i97 or HUECO
        # LA COLA ES LA QUE LA FICHA ESCRIBIÓ con el sentido del auto; si no
        # la trae, la del catálogo sólo cuando es cierta para todo el inciso:
        # la del inciso a) de la fracción I dice «desechó», y ese inciso también
        # es el del auto que ADMITE o tiene por no presentada la demanda.
        # Y LA DEL INCISO e) TAMPOCO (3-oct-2026, revisión de fundamentos): la del
        # catálogo dice «dictada con posterioridad a la sentencia definitiva» y
        # el inciso e) es también el de lo dictado DURANTE el trámite del juicio
        # o del incidente; sin el sentido del auto en la ficha, hueco y aviso.
        # Y SIN FRACCIÓN, TAMPOCO LA DEL CATÁLOGO: la cola de cada inciso es la
        # de su fracción.
        _cola = ""
        if _i97:
            _cola = _valor(_pro.get("cola_97")) or (
                "" if (not _fr97 or (_fr97 == "I" and _i97 in ("a", "e")))
                else _ta.cola_97(_i97, _fr97))
        _datos_bk["cola_97"] = _cola or HUECO
        if not _i97:
            if not _inciso_ya_avisado:
                _avisos_bk.append(
                    f"EL INCISO DEL ARTÍCULO 97{_fr97_aviso} SALE EN HUECO (inciso_97): la "
                    f"ficha de trámite no lo trae. Escríbelo en «Trámite en este "
                    f"tribunal»: es el supuesto que hace procedente la queja.")
        elif not _cola:
            _avisos_bk.append(
                f"QUÉ RESOLVIÓ EL AUTO RECURRIDO SALE EN HUECO en la procedencia "
                f"(inciso {_i97}) "
                + (f"de la fracción {_fr97} " if _fr97 else "")
                + "del artículo 97): complétalo con lo que dice su parte resolutiva.")
        # EL PLAZO VA CON EL INCISO. La suspensión provisional o de plano del
        # amparo indirecto (fr. I, inciso b) se recurre en DOS días (art. 98,
        # fr. I); la de la responsable en el amparo directo (fr. II, inciso b),
        # en cinco (oro_Q: 41 contra 4). Un cómputo con el plazo del otro
        # declara oportuno lo extemporáneo, o al revés.
        _plz = getattr(computo, "plazo", None)
        if _i97 == "b" and _fr97 == "I" and _plz and _plz != 2:
            _avisos_bk.append(
                f"LA QUEJA ES CONTRA LA SUSPENSIÓN PROVISIONAL O DE PLANO "
                f"(artículo 97, fracción I, inciso b) y el cómputo corrió {_plz} "
                f"días: su plazo es de DOS (artículo 98, fracción I, de la Ley de "
                f"Amparo). Elige la excepción «suspensión» y vuelve a generar.")
        if _i97 == "b" and _fr97 == "II" and _plz == 2:
            _avisos_bk.append(
                "LA QUEJA ES CONTRA LO QUE LA RESPONSABLE PROVEYÓ SOBRE LA "
                "SUSPENSIÓN EN UN AMPARO DIRECTO (artículo 97, fracción II, inciso "
                "b) y el cómputo corrió dos días: los dos días son sólo de la "
                "suspensión provisional o de plano del amparo indirecto; aquí el "
                "plazo es de cinco (artículo 98 de la Ley de Amparo).")
    elif _ta.normalizar(tipo_asunto) == "queja":
        # DE DÓNDE SE LEE. `descripcion_acto` cae a menudo al genérico «del
        # auto recurrido», que no dice QUÉ se desechó y deja al inciso sin
        # materia. Los resultandos sí lo describen —los acaba de escribir el
        # modelo con el auto delante— y son cuatro párrafos, no el OCR entero:
        # la heurística de una palabra sólo explota cuando se le da un
        # expediente pegado, y esto no lo es.
        _para_inciso = " ".join([
            str(_datos_bk.get("descripcion_acto") or ""),
            " ".join(str(r.get("texto") or "")
                     for r in (estructura.resultandos or []))[:1200],
        ])
        _i97 = _ta.inciso_97(_para_inciso)
        _datos_bk["inciso"] = _i97 or HUECO
        # Y LA COLA VA CON SU INCISO. Si el inciso no se pudo afirmar, la cola
        # tampoco: emitir la del desechamiento de la demanda junto a un inciso
        # en hueco sería afirmar el hecho y callar el fundamento.
        _datos_bk["cola_97"] = _ta.cola_97(_i97) or HUECO
        if not _i97 and not (_nuevo and _inciso_ya_avisado):
            _avisos_bk.append(
                f"EL INCISO DEL ARTÍCULO 97{_fr97_aviso} SALE EN HUECO: no se "
                "pudo afirmar cuál corresponde a este acto. Escríbelo: es el "
                "supuesto que hace procedente la queja, y uno equivocado se "
                "caza en sesión.")
    # DÓNDE SE DICTÓ EL AUTO RECURRIDO, en la procedencia de la queja (3-oct-2026,
    # Q 337/2026 del banco): la plantilla decía siempre «en el juicio de amparo
    # {expediente}» y perdía «indirecto» en las ocho quejas y el incidente de
    # suspensión en la que lo era, aunque el V I S T O y el resultando lo
    # dijeran bien. Con la ficha, la vía y el incidente; sin ella, como siempre.
    _datos_bk["juicio_de_amparo"] = "juicio de amparo"
    if _tn == "queja" and _nuevo:
        _acto_q = _ficha_t.get("acto") if isinstance(_ficha_t.get("acto"), dict) else {}
        _via_q = (_valor(_pro.get("via_amparo")) or _valor(_acto_q.get("via"))).lower()
        if _via_q not in ("directo", "indirecto"):
            # NI «INDIRECTO» POR OMISIÓN (F3): sin fracción no hay vía que decir.
            _via_q = ("directo" if _fr97 == "II" else "indirecto" if _fr97 == "I" else HUECO)
        _inc_q = bool(_pro.get("incidente")) if "incidente" in _pro else bool(_acto_q.get("incidente"))
        _datos_bk["juicio_de_amparo"] = (
            ("incidente de suspensión relativo al juicio de amparo " if (_inc_q and _via_q == "indirecto")
             else "juicio de amparo ") + _via_q)
    # EL ARTÍCULO LO PONE LA PLANTILLA, NO YO. Las fórmulas del banco ya dicen
    # «por el {responsable}» y «dictado por el {responsable}», así que
    # anteponerle el artículo aquí producía «por el el Juez de Distrito». Se
    # entrega el nombre limpio y la plantilla lo enmarca; donde hace falta
    # artículo —el resolutivo, los efectos— se pone en ese sitio.
    _datos_bk["responsable"] = _sin_articulo(_normalizar_autoridad(_resp))
    # Y LOS DATOS QUE LA PLANTILLA PIDE Y NADIE LLENABA. `{descripcion_acto}`
    # salía como hueco «*********» en la competencia de toda queja: es la única
    # frase que dice CONTRA QUÉ se recurre, y sin ella el considerando primero
    # no se sostiene. Sale del propio acto, que el secretario ya subió.
    # SIEMPRE POR `_descripcion_del_acto` (3-oct-2026): la de la ficha llega
    # cruda —«auto que negó…»— y la plantilla dice «en contra {descripcion_acto}».
    _datos_bk["descripcion_acto"] = _descripcion_del_acto(datos, tipo_asunto)
    # EL EXPEDIENTE DE ORIGEN. La plantilla de «Existencia del acto reclamado»
    # pide `{expediente}` y nadie lo alimentaba: salía «los autos del
    # expediente *********» en el considerando SEGUNDO. Se lee de lo que el
    # modelo YA escribió en los resultandos —que salió del OCR— y no se le
    # vuelve a preguntar: una llamada más es una ocasión más de inventarlo.
    if _pro:
        # LA FICHA MANDA, Y LO QUE NO TRAE NO SE LEE DE LA PROSA (3-oct-2026).
        # El número que cada tipo llama «expediente»: el de origen en el amparo
        # directo (el toca va aparte), el juicio de amparo en la revisión y la
        # queja, el juicio de nulidad en la revisión fiscal.
        _exp_p = {
            "amparo_directo": _pro.get("expediente"),
            "amparo_revision": _pro.get("juicio_amparo") or _pro.get("expediente"),
            "queja": _pro.get("juicio_amparo") or _pro.get("expediente"),
            "revision_fiscal": _pro.get("expediente_tfja") or _pro.get("expediente"),
        }.get(_tn)
        # SIN EL RÓTULO DENTRO DEL NÚMERO: la plantilla ya dice «del expediente
        # {expediente}», y la ficha puede traerlo como «expediente 515/2022».
        _datos_bk["expediente"] = (re.sub(r"(?i)^\s*(?:expediente|exp\.?)\s+", "",
                                          _valor(_exp_p)) or HUECO)
        _datos_bk["fecha_acto"] = _valor(_pro.get("fecha_acto")) or HUECO
        # EL TOCA CON SU RÓTULO, UNA VEZ: «los autos del {toca}» → «del toca
        # familiar 4357/2025». El compositor lo entrega ya rotulado en
        # `toca_en_prosa`; si no, se rotula aquí.
        _t_toca = _valor(_pro.get("toca_en_prosa")) or _valor(_pro.get("toca"))
        if _t_toca:
            _datos_bk["toca"] = (_t_toca if _t_toca.lower().startswith("toca")
                                 else f"toca {_t_toca}")
    elif not str(_datos_bk.get("expediente") or "").strip():
        try:
            import fase_origen as _fo
            _res_txt = " ".join(str(r.get("texto") or "")
                                for r in (estructura.resultandos or []))
            _datos_bk["expediente"] = (_fo.numero_de(_res_txt,
                                                     str(datos.get("numero") or ""))
                                       or HUECO)
            # Y LA FECHA DE LO RECURRIDO, del mismo sitio: la plantilla de
            # procedencia dice «se impugna el auto de {fecha_acto}» y salía en
            # asteriscos justo al lado del inciso que sí se dedujo.
            if not str(_datos_bk.get("fecha_acto") or "").strip().strip("*"):
                _datos_bk["fecha_acto"] = _fo.fecha_de(_res_txt) or HUECO
        except Exception:
            pass
    # LA FRACCIÓN DEL ACUERDO GENERAL ES LA DE CADA CIRCUITO, Y SE DEDUCE.
    # El banco traía escrita la XXII, que reparte la jurisdicción del Vigésimo
    # Segundo, así que un secretario de Mérida nombraba bien a su tribunal y
    # fundaba su competencia en la fracción de otro. Se dejó en hueco visible.
    #
    # Se dejó de más. Lo que no se deduce es del EXPEDIENTE —ahí no consta—,
    # pero sí del TRIBUNAL, que el secretario declara en el formulario: el
    # punto tercero enumera los circuitos en orden, de modo que la fracción es
    # el número del circuito en romanos. El adelanto real del Vigésimo Segundo
    # lo confirma: fracción XXII. Los cuatro proyectos salían con «fracción
    # *********» en el considerando de competencia, que es el primero que se
    # lee, y ese asterisco era evitable.
    #
    # La regla de la casa se respeta: si el tribunal no dice su circuito, sigue
    # saliendo hueco. Y el aviso a los secretarios de otro circuito sigue
    # pidiéndoles revisar la cadena de fundamentos entera antes de firmar,
    # porque la fracción es una pieza de esa cadena, no la cadena.
    # LOS MARCADORES DE LA PLANTILLA DE REVISIÓN, que nadie llenaba: salían como
    # «en materia *********, por el *********» en mitad del considerando de
    # competencia. El juez de distrito ES la autoridad responsable —en una
    # revisión de amparo indirecto lo recurrido es su sentencia— y la materia ya
    # está calculada tres líneas más arriba.
    # CON SU ARTÍCULO, porque la plantilla ya no lo pone: «por el Sala Regional»
    # no es español y la concordancia depende del órgano, no de la frase.
    # EL JUEZ DE DISTRITO ES EL ÓRGANO RECURRIDO, no la responsable del acto
    # (28-sep-2026, AR 631/2025: «dictada en un juicio de amparo indirecto …
    # por la MAGISTRADA…»). Cuando se leyó quién dictó la sentencia recurrida,
    # manda; si no, como antes.
    _org_rec = str(datos.get("organo_recurrido") or "").strip()
    _juez_dist = _org_rec if (_ta.normalizar(tipo_asunto) == "amparo_revision" and _org_rec) else _resp
    _datos_bk.setdefault("juez_distrito", _con_articulo(_juez_dist) or HUECO)
    _datos_bk.setdefault("juzgado", _con_articulo(_juez_dist) or HUECO)
    # «LOCALIZADO» CONCUERDA CON EL ÓRGANO QUE LA PLANTILLA NOMBRA, y su
    # género lo da su artículo (3-oct-2026). Se calculaba con una lista de
    # palabras sobre la responsable: «por el Jueza Tercero… localizado» en la
    # queja, «la Sala Regional… localizado» en la revisión fiscal. El artículo
    # de `_con_articulo` ya sabe de jueza, magistrada, sala y los sufijos
    # femeninos, y el órgano es el mismo que va en {juzgado}.
    _datos_bk.setdefault("concordancia",
                         "localizada" if _con_articulo(_juez_dist).lower().startswith("la ")
                         else "localizado")
    # EL TRIBUNAL EN LA PROSA, COMO NOMBRE (3-oct-2026). La competencia dice
    # «Este {tribunal}» y lo recibía crudo: tecleado en versales salía «Este
    # TERCER TRIBUNAL COLEGIADO…» (AD_tribunal_versales), y con su artículo,
    # «Este el…». El proemio ya lo pasaba por `_nombre_de_organo`; la
    # competencia, que es lo primero que se lee, no.
    _datos_bk["tribunal"] = (_sin_articulo(_nombre_de_organo(datos.get("tribunal")))
                             or str(datos.get("tribunal") or ""))
    # EL SUPLETORIO DE LA SEDE (3-oct-2026, David): el Código Federal de
    # Procedimientos Civiles en toda la república; el Código Nacional sólo en
    # la Ciudad de México. Ver `tipos_asunto.supletorio`.
    _sup = (_pro.get("supletorio") if isinstance(_pro.get("supletorio"), dict)
            and _pro["supletorio"].get("documentales") else
            _ta.supletorio(str(datos.get("tribunal") or ""), str(datos.get("ciudad") or "")))
    _datos_bk.setdefault("supletorio_documentales", _sup.get("documentales") or HUECO)
    # VACÍO = RECURRE LA QUEJOSA (convención del encargo). `setdefault` no
    # llenaba la clave cuando venía vacía, y las fórmulas del banco que dicen
    # «{recurrente}, interpuso recurso de revisión» salían sin sujeto
    # (28-sep-2026).
    if not str(_datos_bk.get("recurrente") or "").strip():
        _datos_bk["recurrente"] = str(datos.get("quejoso") or "").strip() or HUECO
    _datos_bk.setdefault(
        "fraccion_acuerdo",
        str(datos.get("fraccion_acuerdo") or "").strip()
        or _bk.fraccion_del_acuerdo(str(datos.get("tribunal") or ""))
        or HUECO)
    _datos_bk.setdefault("fecha_acto", str(datos.get("fecha_acto") or "").strip()
                         or HUECO)
    # LO RECLAMADO EN EL AMPARO DIRECTO, POR SU CLASE (3-oct-2026, AD 349 y 552
    # del banco): «una sentencia definitiva» estaba fija y el V I S T O decía
    # «la resolución dictada…». Con la ficha, la resolución que puso fin al
    # juicio y el laudo se llaman así en la competencia, y `_con_su_clase`
    # cambia «la sentencia reclamada» en la existencia, la legitimación, la
    # oportunidad y la dispensa.
    _clase_ad = (_clase_del_acto_ad(_pro, _ficha_t)
                 if (_nuevo and _tn == "amparo_directo") else "sentencia")
    if _clase_ad == "resolucion":
        # «QUE PUSO FIN AL JUICIO» SÓLO SI LO CONCLUYÓ SIN RESOLVER EL FONDO
        # (3-oct-2026, cuarta ronda, E4; AD 552/2024 del banco: la apelación de
        # un sumario hipotecario resuelta en el fondo salió «una resolución que
        # puso fin al juicio»). El artículo 170, fracción I, de la Ley de
        # Amparo: «por resoluciones que pongan fin al juicio, las que sin
        # decidirlo en lo principal lo den por concluido» (texto local). Que la
        # ficha la llame «resolución» da su nombre, no su naturaleza: por
        # omisión, la fórmula neutra del engrose del AD 349, «una resolución
        # definitiva en materia X»; la otra, sólo si el sentido lo dice, y con
        # aviso.
        _fin = _pone_fin_al_juicio(_pro, _ficha_t)
        if _fin:
            _datos_bk["objeto"] = ("una resolución que puso fin al juicio en materia " + _mat
                                   if _mat else "una resolución que puso fin al juicio")
            _avisos_bk.append(
                f"LA COMPETENCIA LLAMA A LO RECLAMADO «UNA RESOLUCIÓN QUE PUSO FIN AL JUICIO» "
                f"porque el sentido que trae la ficha es {_fin} (acto.sentido): compruébalo. El artículo 170, "
                f"fracción I, de la Ley de Amparo entiende por tales «las que sin decidirlo en lo "
                f"principal lo den por concluido»; si la responsable resolvió el fondo, es «una "
                f"resolución definitiva».")
        else:
            _datos_bk["objeto"] = ("una resolución definitiva en materia " + _mat
                                   if _mat else "una resolución definitiva")
    elif _clase_ad == "laudo":
        _datos_bk["objeto"] = f"un laudo en materia {_mat}" if _mat else "un laudo"
    _datos_bk.setdefault("objeto", (f"una sentencia definitiva en materia {_mat}"
                                    if _mat else "una sentencia definitiva"))
    _huecos_bk = []

    def _del_banco(ident, respaldo, variante=""):
        """La frase del oficio si el banco la tiene; si no, la del modelo.

        LOS HUECOS SE NOMBRAN (3-oct-2026). Un marcador sin valor, o con el
        hueco puesto a propósito porque la ficha no trae el dato, queda
        apuntado con su apartado; al final sale un aviso por apartado que dice
        QUÉ DATO falta, no qué marcador."""
        t, faltan = _bk.texto_de(tipo_asunto, ident, _datos_bk, variante=variante)
        if variante and not t:
            t, faltan = _bk.texto_de(tipo_asunto, ident, _datos_bk)
        if t:
            _pl = (_bk.variante_de(tipo_asunto, ident, variante) if variante else "") \
                or str(_bk.apartado(tipo_asunto, ident).get("plantilla") or "")
            _en_hueco = list(faltan) + [
                m for m in _bk._RX_MARCA.findall(_bk._a_marcadores(_pl))
                if str(_datos_bk.get(m) or "").strip() == HUECO]
            for m in _en_hueco:
                if (ident, m) not in _huecos_bk:
                    _huecos_bk.append((ident, m))
            return t
        return respaldo


    # LA QUEJA DE LA FRACCIÓN II tiene su propia cadena (3-oct-2026): el acto
    # es de la responsable en un amparo directo radicado en este tribunal, no
    # de un juez de distrito en un indirecto.
    _var_q = "qj-c1-fraccion-ii" if (_tn == "queja" and _fr97 == "II") else ""
    _comp = _del_banco("competencia", estructura.competencia, variante=_var_q)
    # SIN FRACCIÓN, NI LA I NI «INDIRECTO» (F3): la cadena de la fracción I con
    # la fracción y la vía en hueco, y el dato apuntado para su aviso.
    if _fr97_en_hueco:
        _comp = _fraccion_97_en_hueco(_comp, _huecos_bk, "competencia")
    # QUÉ SE RECURRE EN LA REVISIÓN: la ficha lo dice si lo leyó del papel
    # (sentencia, interlocutoria de suspensión, auto de sobreseimiento); si no,
    # el proemio de la recurrida, como antes.
    _clase_ficha = (_valor(_pro.get("clase_recurrida"))
                    or (_valor(_ficha_t.get("clase_recurrida")) if _ficha_t else ""))
    _clase_ar = (_ta.clase_recurrida_de(str(datos.get("acto") or "")[:1500], _clase_ficha)
                 if _tn == "amparo_revision" else "")

    def _con_su_clase(t: str) -> str:
        """«la sentencia recurrida» → lo que de verdad se recurre (3-oct-2026,
        bandera). En el incidente en revisión y en la revisión del auto que
        sobresee fuera de audiencia, el proyecto llamaba «sentencia» a una
        interlocutoria o a un auto en la oportunidad, la dispensa, el cierre y
        el resolutivo (oro_AR: «se confirma la [sentencia|interlocutoria|auto]
        recurrid[a/o]»).

        Y EN EL AMPARO DIRECTO, «la sentencia reclamada» → «la resolución
        reclamada» o «el laudo reclamado» cuando la ficha dice que lo reclamado
        es eso (tercera ronda; AD 349 y 552 del banco)."""
        if _clase_ad in ("resolucion", "laudo"):
            _n = "la resolución reclamada" if _clase_ad == "resolucion" else "el laudo reclamado"
            t = re.sub(r"\bla\s+sentencia\s+reclamada\b", _n, t or "")
            t = re.sub(r"\bLa\s+sentencia\s+reclamada\b", _n[:1].upper() + _n[1:], t)
            t = re.sub(r"\bSentencia\s+reclamada\b",
                       "Resolución reclamada" if _clase_ad == "resolucion" else "Laudo reclamado", t)
            return _contraer(t)
        if not (_rige_pt and _clase_ar in ("interlocutoria_suspension", "auto_sobreseimiento")):
            return t
        _n = ("la interlocutoria recurrida" if _clase_ar == "interlocutoria_suspension"
              else "el auto recurrido")
        t = re.sub(r"\bla\s+sentencia\s+recurrida\b", _n, t or "")
        t = re.sub(r"\bLa\s+sentencia\s+recurrida\b", _n[:1].upper() + _n[1:], t)
        return _contraer(t)
    if _ta.normalizar(tipo_asunto) == "amparo_revision" and (_comp or "").strip():
        # LA DEMANDA DE LA FICHA TAMBIÉN DICE SI SE IMPUGNÓ UNA LEY (3-oct-2026,
        # AR 208 y 72/2025 del banco): el juzgado concedió contra la Ley de
        # Hacienda del Estado y recurrió el Gobernador; la competencia salió con
        # la cadena ordinaria y sin el aviso de la delegación, porque sólo se
        # le daban los antecedentes y la prosa de los resultandos. Las
        # autoridades y los actos de la demanda, anclados al papel, van también.
        _dem_f = _ficha_t.get("demanda") if isinstance(_ficha_t.get("demanda"), dict) else {}
        _auts_f = [_valor(a_) for a_ in (list(_pro.get("autoridades") or [])
                                          or list(_dem_f.get("autoridades") or []))
                   if isinstance(a_, str) and _valor(a_)] if _nuevo else []
        _actos_f = [_valor(a_) for a_ in (_dem_f.get("actos") or [])
                    if isinstance(a_, str) and _valor(a_)] if _nuevo else []
        _comp, _av_comp = _ta.competencia_revision(
            _comp, str(datos.get("acto") or "")[:1500],
            " ".join([str(datos.get("antecedentes") or ""),
                      " ".join(str(r.get("texto") or "")
                               for r in (estructura.resultandos or []))]
                     + _actos_f + _auts_f),
            clase=_clase_ficha)
        _avisos_bk.extend(_av_comp)
        if _nuevo:
            _avisos_bk.extend(_aviso_delegacion_scjn(
                _auts_f, _actos_f, _av_comp,
                # «mixto» a secas no dice si concedió: se mira su detalle.
                resolvio=" ".join(
                    [_valor((_ficha_t.get("acto") or {}).get(k_))
                     for k_ in ("resolvio_mixto", "resolvio")
                     if isinstance(_ficha_t.get("acto"), dict)]
                    + [_valor(datos.get("resolvio_a_quo"))]),
                papel=(_valor(_ficha_t.get("caracter")) or _valor(datos.get("papel_recurrente"))),
                clase=_clase_ar))
    # AMPARO DIRECTO CONTRA UN JUICIO ORAL MERCANTIL (30-sep-2026): la
    # coletilla del banco, pegada al final del mismo párrafo, como la pegan los
    # engroses (ver `coletilla_oral_mercantil`). Sin única instancia o sin la
    # vía en los autos, vacía y el párrafo queda como estaba.
    if (_comp or "").strip() and "1390" not in _comp:
        _ante_c = antecedentes if isinstance(antecedentes, str) else " ".join(
            str(x) for x in (antecedentes or []) if isinstance(x, str))
        _col = coletilla_oral_mercantil(tipo_asunto, {
            "acto": datos.get("acto"),
            "antecedentes": " ".join([str(datos.get("antecedentes") or ""), _ante_c])})
        if _col:
            _comp = _comp.rstrip() + " " + _col
    if _clase_ad == "laudo" and (_comp or "").strip():
        # «un laudo…, dictada por» no concuerda: el participio va con el laudo.
        _comp = re.sub(r"(un\s+laudo(?:\s+en\s+materia\s+[^,]+)?),\s+dictada\s+por",
                       r"\1, dictado por", _comp)
    if _clase_ad in ("resolucion", "laudo"):
        _comp = _con_su_clase(_comp)
    if _nuevo and _tn == "queja" and (_comp or "").strip():
        # LA COMA QUE CIERRA EL INCISO (3-oct-2026; 5 de 8 engroses de queja del
        # banco y la propia procedencia de este documento): «97, fracción I,
        # inciso a), de la Ley de Amparo». Con la ficha; el camino de siempre
        # conserva su plantilla.
        # Y CON LA FRACCIÓN EN HUECO (F3), igual.
        _comp = re.sub(r"(97,\s+fracci[óo]n\s+(?:I{1,2}|\*{9}),\s+inciso\s+[^)\s]{1,12}\))"
                       r"\s+de\s+la\s+Ley\s+de\s+Amparo",
                       r"\1, de la Ley de Amparo", _comp)
    if (_comp or "").strip():
        con_apartados.append((_bk.rotulo_de(tipo_asunto, "competencia", "Competencia."),
                              (lambda c: lambda p: _texto_en(p, c))(_comp)))
    # LA EXISTENCIA DEL AMPARO DIRECTO, CON EL TOCA Y EL EXPEDIENTE APARTE
    # (3-oct-2026, bandera). La plantilla general dice «los autos del
    # expediente {expediente}» y el expediente se leía de la prosa: en el AD_xxii
    # el resultando decía «toca civil 374/2024… expediente 905/2023» y la
    # existencia salió «los autos del expediente 374/2024», el toca. Con la ficha,
    # si hay toca, la variante del corpus que los nombra a los dos (`ad-c2-civil`,
    # 32 de 86); si no hay toca (única instancia), la general con el expediente.
    _var_exi = ("ad-c2-civil" if (_pro and _tn == "amparo_directo"
                                  and _valor(_datos_bk.get("toca"))) else "")
    # EL CÓDIGO SUPLETORIO YA NO VA ESCRITO EN LA PLANTILLA: lo pone
    # `{supletorio_documentales}`, según la sede (`tipos_asunto.supletorio`).
    # Este comentario decía que eran «el 129 y el 202 del Código Federal… y no
    # cambian nunca»; cambiaron dos veces en una semana: al Nacional en toda la
    # república (27-sep) y, por decisión de David (3-oct), al Federal salvo en
    # la Ciudad de México.
    # LA REVISIÓN NO LA PIDE (C1, 3-oct-2026): ni al banco ni al modelo, para
    # que ningún hueco suyo llegue a los avisos.
    _exi = (_del_banco("existencia", estructura.existencia, variante=_var_exi)
            if _tn != "amparo_revision" else "")
    # SE COMPONE AUNQUE EL MODELO HAYA ESCRITO ALGO. Escribió esto: «se acredita
    # con el informe justificado rendido por LAS AUTORIDADES RESPONSABLES y con
    # las constancias que integran los autos del juicio de amparo indirecto DE
    # ORIGEN» —sin decir qué órgano, sin el número, y sin los preceptos que dan
    # valor probatorio a esas documentales—. La fórmula es fija y los datos los
    # tiene el compositor: no hay nada que preguntar.
    #
    # ═══ LA REVISIÓN YA NO LLEVA EXISTENCIA (C1, 3-oct-2026) [siempre] ════
    # Aquí se componía «La existencia de la sentencia recurrida está acreditada
    # con los autos originales del juicio de amparo indirecto…, que remitió el
    # Juzgado… en términos del artículo 89 (o 90) de la Ley de Amparo», con su
    # aviso «EL CONSIDERANDO DE EXISTENCIA SALE CON HUECO». David: «el
    # considerando de existencia ya no es necesario porque ya viene en la
    # sentencia recurrida; no es usual ni necesario que lo reproduzcamos». El
    # esqueleto de la revisión ya no lo declara (`ESQUELETO`) y en su lugar va
    # la procedencia del recurso, más abajo (`tipos_asunto.procedencia_revision`).
    if _clase_ad in ("resolucion", "laudo"):
        _exi = _con_su_clase(_exi)
    if esq["existencia"] and (_exi or "").strip():
        if _sup.get("aviso") and _sup["aviso"] not in _avisos_bk:
            _avisos_bk.append(_sup["aviso"])
        con_apartados.append((_bk.rotulo_de(tipo_asunto, "existencia",
                                            "Existencia del acto reclamado."),
                              (lambda c: lambda p: _texto_en(p, c))(_exi)))

    # Legitimación y oportunidad, con LA TABLA detrás.
    from fase0_oportunidad import parrafo_oportunidad
    # EL FUNDAMENTO Y EL VOCABULARIO SALEN DEL CATÁLOGO. Antes decía siempre
    # «el precepto 17 del mencionado ordenamiento», que es el del amparo: una
    # queja se fundaba en el artículo del amparo y una revisión fiscal, en la
    # Ley de Amparo cuando la suya es la LFPCA.
    # QUIÉN RECURRE DECIDE LA FRACCIÓN DEL 31 (3-oct-2026): si es una
    # autoridad, su notificación por oficio surte desde que queda hecha (fr. I),
    # no al día siguiente como la de los particulares (fr. II).
    _papel_op = str(datos.get("papel_recurrente") or "").strip().lower()
    # EL CARÁCTER DE QUIEN RECURRE, DE LA FICHA (3-oct-2026, bandera): si el
    # encargo no lo dijo y la ficha de trámite sí, manda la ficha.
    if not _papel_op and _ficha_t and _tn in ("amparo_revision", "queja"):
        _car_f = str(_ficha_t.get("caracter") or "").strip().lower()
        if _car_f in ("quejoso", "autoridad", "tercero", "ministerio_publico"):
            _papel_op = _car_f
    # EL PRECEPTO DEL SURTIMIENTO (3-oct-2026, bandera). En el amparo directo
    # lo fija la ley del acto: si el secretario lo escribió en la ficha
    # («fundamento_surtimiento»), va en el sitio del hueco; si no, y la ley del
    # acto es federal y lo dice para toda la república (Código de Comercio en
    # lo mercantil, LFPCA ante el TFJA), ésa, con un aviso que pide comprobarlo.
    _f_surte_decl, _av_surte_nac = "", ""
    if _rige_pt:
        _f_surte_decl = (_valor(_pro.get("fundamento_surtimiento"))
                         or _valor((_ficha_t or {}).get("fundamento_surtimiento")))
        if not _f_surte_decl:
            try:
                import fase0_oportunidad as _f0_sn
                # LO MERCANTIL REGISTRADO COMO CIVIL (3-oct-2026, integración;
                # AD 456/2025 del banco): el expediente en prosa («juicio oral
                # mercantil 625/2024») y el encabezado o los antecedentes lo dicen
                # aunque la materia de la ficha sea «civil».
                _ant_sn = antecedentes if isinstance(antecedentes, str) else " ".join(
                    str(x) for x in (antecedentes or []) if isinstance(x, str))
                # LA SEDE DECIDE LO AGRARIO (C7, 3-oct-2026; David: «si va con el
                # Código Nacional, y no el Federal, entonces hay que adecuar al
                # Código Nacional»): en la Ciudad de México la regla del Código
                # Nacional ya trae su precepto («»); fuera de ella, o sin sede,
                # el 321 del CFPC con el aviso del transitorio. La sede es la del
                # colegiado: la de la ficha si la trae, la del encargo si no.
                _sede_f = (_ficha_t or {}).get("sede")
                _sede_f = _sede_f if isinstance(_sede_f, dict) else {}
                _sede_sn = {k_: (_valor(_sede_f.get(k_)) or str(datos.get(k_) or "").strip())
                            for k_ in ("tribunal", "ciudad")
                            if _admite(_f0_sn.surtimiento_nacional, k_)}
                _f_surte_decl, _av_surte_nac = _f0_sn.surtimiento_nacional(
                    tipo_asunto, _valor(_pro.get("materia")) or str(datos.get("materia") or ""),
                    str(datos.get("responsable") or ""),
                    str(getattr(getattr(computo, "regla", None), "clave", "") or ""),
                    expediente=" ".join(x for x in (_valor(_pro.get("expediente_en_prosa")),
                                                     _valor(_pro.get("toca_en_prosa"))) if x),
                    contexto=" ".join([str(datos.get("encabezado") or ""), _ant_sn[:4000]]),
                    **_sede_sn)
            except Exception:
                _f_surte_decl, _av_surte_nac = "", ""
    # LA FORMA DE NOTIFICACIÓN QUE NO CONSTA NO SE AFIRMA (3-oct-2026, quinta
    # ronda, F1; Q 335, 229, 261 y 342/2025 y AD 274 y 335/2025 del banco: «se
    # notificó… de manera personal» con la forma vacía en la ficha y sin aviso).
    # El formulario trae «personal» por omisión, y con la autoridad que recurre,
    # «oficio» (D2): ninguno de los dos es un dato del papel. Con la ficha, si
    # la fuente de `forma_notificacion` es la omisión —o la ficha no la trae— y
    # el cómputo corrió con esa regla, el párrafo dice la fecha y cuándo surtió
    # efectos, sin la forma (como los engroses), y un aviso pide comprobarla.
    _forma_omision = _forma_de_omision(_ficha_t, computo) if _nuevo else ""
    _kw_op = {}
    if _forma_omision:
        if _admite(parrafo_oportunidad, "forma_consta"):
            _kw_op["forma_consta"] = False
        if _admite(parrafo_oportunidad, "fuente_forma"):
            _kw_op["fuente_forma"] = _forma_omision
    _op = parrafo_oportunidad(
        computo,
        _ta.plazo_de(tipo_asunto, "").get("fundamento") or "artículo 17 de la Ley de Amparo",
        tipo_asunto, papel=_papel_op, sin_precepto_en_hueco=_rige_pt,
        fundamento_surtimiento=_f_surte_decl,
        # LOS INHÁBILES ENTRE LA NOTIFICACIÓN Y EL INICIO DEL PLAZO (3-oct-2026,
        # integración): sólo en el camino nuevo (bandera Y ficha). Sin esto el
        # párrafo saltaba del 18 de diciembre al 2 de enero sin decir por qué
        # (Q 24/2026) o callaba el 21 de marzo (AD 335/2025).
        con_previos=bool(_rige_pt and _nuevo), **_kw_op)
    if _forma_omision:
        # Si la pieza del párrafo aún no sabe omitirla, se quita aquí: la forma
        # va siempre entre la fecha de la notificación y «y surtió efectos».
        _op = _sin_forma_de_notificacion(_op, computo)
        _av_forma = _aviso_forma_omision(computo, tipo_asunto, _papel_op)
        if _av_forma and not _clave_ya_avisada(("forma_notificacion",),
                                               _avisos_de_fuera + list(_avisos_bk)):
            _avisos_bk.append(_av_forma)
    _op = _con_su_clase(_op)
    if _av_surte_nac and _f_surte_decl and _f_surte_decl.split(" ", 1)[-1] in _op:
        _avisos_bk.append(_av_surte_nac)
    # EL AVISO SIN PRECEPTO (revisión de normas y front, 3-oct-2026): la regla
    # del Código Nacional en la Ciudad de México ya trae el suyo, así que
    # `surtimiento_nacional` da («», aviso del transitorio Tercero). Se escribe
    # si el párrafo cuenta con esa regla, y una sola vez.
    elif (_av_surte_nac and not _f_surte_decl
          and str(getattr(getattr(computo, "regla", None), "clave", "") or "").startswith("cnpcf_")
          and _av_surte_nac not in (_avisos_de_fuera + list(_avisos_bk))):
        _avisos_bk.append(_av_surte_nac)
    # EL SURTIMIENTO SIN PRECEPTO SE AVISA. En el amparo directo con
    # notificación personal lo fija la ley del acto, que el catálogo no trae:
    # el considerando dice «conforme a la ley del acto» y quien firma escribe el
    # artículo. Ver `fase0_oportunidad.aviso_fundamento`.
    try:
        from fase0_oportunidad import aviso_fundamento as _av_fund
        # Y EL AVISO TAMPOCO REPITE LA FORMA QUE NO CONSTA (F1): «el
        # considerando dice que la notificación de manera personal surtió…».
        _kw_af = {k_: v_ for k_, v_ in _kw_op.items() if _admite(_av_fund, k_)}
        _avf = _av_fund(computo, tipo_asunto, _papel_op, en_hueco=_rige_pt,
                        fundamento_surtimiento=_f_surte_decl, **_kw_af)
        if _avf and _forma_omision:
            _avf = _sin_forma_de_aviso(_avf, computo)
        if _avf:
            _avisos_bk.append(_avf)
    except Exception:
        pass

    # LA LEGITIMACIÓN VA PRIMERO, y sin ella el párrafo del cómputo abre con
    # «Igualmente,» sin nada a lo que enlazar. Se compone: quién interpuso, en
    # qué carácter, por qué precepto de SU vía y por qué le perjudica.
    # LA AUTORIDAD DEMANDADA DE LA REVISIÓN FISCAL, si la ficha la trae: la
    # unidad que recurre lleva su defensa jurídica (art. 63 LFPCA).
    _aut_dem = (_valor(_pro.get("autoridad_demandada"))
                or _valor((_ficha_t or {}).get("autoridad_demandada"))
                or _valor(datos.get("autoridad_demandada")))
    # LA QUEJA, CON LA FRACCIÓN DEL 5o. DE QUIEN RECURRE (3-oct-2026): I la
    # quejosa, II la autoridad, III la tercera, IV el Ministerio Público. Sin
    # papel, la del quejoso sólo si consta que recurre él (`recurrente` vacío
    # es «el mismo», o es la misma parte); si no, hueco y aviso. Y si recurre
    # otra parte, el párrafo es suyo.
    _parte_leg = str(datos.get("quejoso") or "")
    _rep_leg = str(datos.get("representante") or "")
    _fig_leg = str(datos.get("figura_representante") or "")
    _moral_leg = datos.get("quejoso_moral")
    _recurre_q = None
    # CON LA FICHA, QUIEN RECURRE SALE DE LA FICHA (3-oct-2026, tercera ronda;
    # SPEC §0.3: los considerandos leen sus datos de la ficha, no del encargo).
    _leg_de_ficha = _nuevo and bool(
        _valor(_ficha_t.get("promovente"))
        or (_tn == "revision_fiscal" and _valor(_pro.get("recurrente_unidad"))))
    if _tn == "queja" and not _leg_de_ficha:
        _rec_q = str(datos.get("recurrente") or "").strip()
        try:
            import promovente as _pv_q
            _recurre_q = (not _rec_q) or _pv_q.misma_parte(_rec_q, _parte_leg)
        except Exception:
            _recurre_q = not _rec_q
        if _rec_q and not _recurre_q:
            try:
                import promovente as _pv_q2
                _sep_q = _pv_q2.separar(_rec_q)
            except Exception:
                _sep_q = {"parte": _rec_q, "representante": "", "figura": "", "moral": None}
            _parte_leg = _sep_q.get("parte") or _rec_q
            _rep_leg = _sep_q.get("representante") or ""
            _fig_leg = _sep_q.get("figura") or ""
            _moral_leg = _sep_q.get("moral")
        if not _ta.fraccion_5o_de(_papel_op, _recurre_q) and _parte_leg.strip():
            _avisos_bk.append(
                "LA FRACCIÓN DEL ARTÍCULO 5o. EN LA LEGITIMACIÓN DE LA QUEJA VA EN HUECO: "
                "no consta en qué carácter recurre " + _parte_leg.strip() + " —quejosa "
                "(fr. I), autoridad responsable (fr. II), tercera interesada (fr. III) o "
                "Ministerio Público (fr. IV)—. Dilo en el encargo («quién recurre») y "
                "vuelve a generar.")
    _leg = _ta.legitimacion_de(
        tipo_asunto, _parte_leg, _rep_leg, HUECO,
        figura=_fig_leg,
        moral=_moral_leg,
        papel=(_papel_op if _tn in ("queja",) else ""),
        autoridad_demandada=_aut_dem, recurre_el_quejoso=_recurre_q)
    # QUIEN INTERPUSO LA REVISIÓN, CON SU CARÁCTER (28-sep-2026, AR 631/2025):
    # si no es la quejosa, el párrafo es suyo —«interpuesto por» la tercera
    # interesada o la autoridad— y su fundamento no es el 6o. (el de quien
    # promueve el amparo).
    _rec_leg = str(datos.get("recurrente") or "").strip()
    _papel_leg = str(datos.get("papel_recurrente") or "").strip().lower()
    if (_ta.normalizar(tipo_asunto) == "amparo_revision" and _rec_leg
            and _papel_leg in ("tercero", "autoridad")):
        try:
            import promovente as _pv_l
            _sep_l = _pv_l.separar(_rec_leg)
        except Exception:
            _sep_l = {"parte": _rec_leg, "representante": "", "figura": "", "moral": None}
        _leg = _ta.legitimacion_de(
            tipo_asunto, _sep_l.get("parte") or _rec_leg,
            _sep_l.get("representante") or "", HUECO,
            figura=_sep_l.get("figura") or "", moral=_sep_l.get("moral"),
            papel=_papel_leg) or _leg
    # ═══ LA LEGITIMACIÓN CON LA FICHA (3-oct-2026, tercera ronda) ══════════
    # Tomaba el dato crudo del encargo y lo partía con `promovente.separar`:
    # salía «interpuesto por el GOBERNADOR DEL ESTADO DE QUERÉTARO (AUTORIDAD
    # RESPONSABLE)» sin el delegado que la ficha sí traía (AR 208 y 72/2025),
    # la queja de la autorizada perdía su representación cuando quejoso y
    # recurrente se escribían distinto (Q 229/2026), y la revisión fiscal
    # afirmaba que «el Jefe del Departamento X» era la unidad de defensa
    # jurídica «del Jefe del Departamento X» (RF 21/2025 y 6/2026).
    if _nuevo and (_leg_de_ficha or _tn == "amparo_directo"):
        _leg_f, _av_leg_f = _legitimacion_con_la_ficha(
            _tn, _ficha_t, _pro, datos, _papel_op, _aut_dem)
        _avisos_bk.extend(_av_leg_f)
        if _leg_f:
            _leg = _leg_f
    if _clase_ad in ("resolucion", "laudo"):
        _leg = _con_su_clase(_leg)

    def _legitimacion(p):
        if _leg:
            _texto_en(p, _leg)
            parrafo(doc, _op)
        else:
            _texto_en(p, _op)
            _avisos_bk.append(
                "EL CONSIDERANDO PROMETE «LEGITIMACIÓN Y OPORTUNIDAD» Y SÓLO "
                "TRAE LA OPORTUNIDAD: no consta quién interpuso el recurso, "
                "así que no se puede afirmar que esté legitimado. Escríbelo, o "
                "el párrafo del cómputo abre con un «Igualmente» que no enlaza "
                "con nada.")
        # La tabla va en los cuatro tipos. En el corpus casi no aparece
        # —el secretario la dibuja a mano y cuesta—, pero eso mide lo que hoy
        # es caro hacer, no lo que sobra: la máquina tiene el calendario.
        # EN UN PLAZO DE AÑOS NO HAY CALENDARIO QUE DIBUJAR: ocho años de
        # recuadros no enseñan nada; el párrafo dice de qué fecha a qué fecha.
        if esq["tabla_computo"] and not getattr(computo, "plazo_anios", 0):
            # EL CALENDARIO PRIMERO, EL MAPA DESPUÉS. El calendario enseña qué
            # fue cada día —con su palabra dentro del recuadro y su leyenda—;
            # el mapa, el recorrido de la notificación a la presentación y el
            # resultado. Las fechas en letra están en el párrafo de arriba,
            # que es lo que se copia al engrose (David, 25-sep-2026).
            calendario_computo(doc, computo, tipo_asunto, _papel_op, _f_surte_decl,
                               forma_consta=not _forma_omision)
            mapa_computo(doc, computo, fecha_en_letra, tipo_asunto, _papel_op, _f_surte_decl,
                         forma_consta=not _forma_omision)

    _legit = (_bk.rotulo_de(tipo_asunto, "legitimacion", esq["legitimacion"]),
              _legitimacion)

    # EL ORDEN ENTRE PROCEDENCIA Y LEGITIMACIÓN LO FIJA EL CATÁLOGO, no este
    # archivo. En la queja el corpus resuelve primero si el recurso procede
    # —21 competencias, 20 procedencias, 16 legitimaciones, en ese orden— y el
    # adelanto real de la QC 259/2025 hace lo mismo: SEGUNDO Procedencia,
    # TERCERO Legitimación. Aquí salía al revés porque el orden estaba escrito
    # a mano, y con razón: en el amparo directo la legitimación va primero.
    # Es la misma pregunta de siempre —¿quién manda, el código o lo medido?—
    # y la respuesta no cambia.
    _cons = [c for c, _ in
             (_ta.estructura_de(tipo_asunto).get("considerandos") or [])]

    def _puesto(clave: str) -> int:
        for i, c in enumerate(_cons):
            if clave in c.lower():
                return i
        return 99

    _procedencia_primero = _puesto("procedencia") < _puesto("legitim")
    if not _procedencia_primero:
        con_apartados.append(_legit)

    # PROCEDENCIA NO ES UN CONSIDERANDO PROPIO cuando hay «Existencia del acto
    # reclamado»: medido, es su ALTERNATIVA —3 de 26, en asuntos venidos de
    # juez y no de sala—, no un apartado más. Emitirlos los dos corría todo un
    # ordinal y dejaba el Estudio en SÉPTIMO donde el corpus lo tiene SEXTO.
    # PROCEDENCIA sólo donde el corpus la tiene como apartado propio —queja
    # (14 de 20) y revisión fiscal (16 de 28)— o en el amparo cuando sustituye
    # a «Existencia del acto reclamado». En revisión civil aparece en 2 de 31:
    # emitirla por defecto ahí corría un ordinal contra la medida. YA NO (C1,
    # 3-oct-2026): la revisión deja la existencia y lleva su procedencia en el
    # mismo sitio, así que el ordinal del estudio no se corre.
    # LA PROCEDENCIA DE LA REVISIÓN FISCAL SE MOTIVA DE OFICIO. Salía «El
    # juicio es procedente y no se advierte causa de improcedencia» —fórmula
    # del amparo, y encima llamando JUICIO a un recurso— porque este apartado
    # tomaba SIEMPRE el texto del modelo y nunca el del banco, aunque el banco
    # tenga medida la suya. Es la tercera vez que aparece el mismo patrón: una
    # fórmula del corpus que existe y es inalcanzable.
    #
    # Y en revisión fiscal ni siquiera basta la del banco: el 63 de la LFPCA
    # obliga al Colegiado a decir POR QUÉ procede, y la vía más común es la
    # cuantía. Eso es aritmética, no criterio, así que se calcula.
    _proc = estructura.procedencia or ""
    if _ta.normalizar(tipo_asunto) == "revision_fiscal":
        try:
            import fase_procedencia_rf as _pf
            _fuente = " ".join([
                str(datos.get("antecedentes") or ""),
                " ".join(str(r.get("texto") or "")
                         for r in (estructura.resultandos or [])),
                str(datos.get("acto") or "")])
            # El año es el de la resolución recurrida. Se toma el de su
            # notificación, que es el dato duro que el secretario tecleó: entre
            # emisión y notificación median días, no años, salvo en el cambio
            # de ejercicio —y ahí el aviso lo dirá, porque la cifra saldrá.
            _anio = getattr(computo, "notificacion", None)
            _anio = _anio.year if _anio else 0
            # EL UMBRAL LO DECIDE LA FECHA DE LA SENTENCIA RECURRIDA (reforma
            # LFPCA DOF 09-06-2026: 27,000 UMA para las dictadas desde el 10 de
            # junio). Se lee la del PDF y, si no, la de la prosa; si no se lee,
            # decide la notificación, con aviso en la frontera.
            _f_sent = None
            if _pro:
                # CON LA FICHA, LA FECHA ES LA SUYA, Y NUNCA LA DE LA PROSA
                # (3-oct-2026, bandera): `fecha_de` sobre los resultandos se
                # llevó en el RF_yucatan la del auto de Presidencia. Sin fecha
                # en la ficha, decide la notificación, con el aviso de frontera
                # de siempre.
                try:
                    import datetime as _dt_pf
                    _iso = _valor(_pro.get("fecha_acto_iso"))
                    _f_sent = (_dt_pf.date.fromisoformat(_iso[:10]) if _iso else None) \
                        or _pf.fecha_de_letra(_valor(_pro.get("fecha_acto")))
                except Exception:
                    _f_sent = None
            else:
                try:
                    import fase_origen as _fo_pf
                    _f_sent = (_pf.fecha_de_letra(str(datos.get("fecha_origen") or ""))
                               or _pf.fecha_de_letra(_fo_pf.fecha_de(_fuente)))
                except Exception:
                    _f_sent = None
            # LA FRACCIÓN Y LA CUANTÍA DE LA FICHA (3-oct-2026, bandera): el
            # secretario dice por qué fracción del 63 procede y la cuantía en
            # pesos; mandan sobre lo que se lea de la prosa. La III necesita
            # saber quién dictó la resolución impugnada.
            _fr63 = _cuantia63 = _aut63 = ""
            if _pro or _ficha_t:
                _fr63 = _valor(_pro.get("fraccion_63")) or _valor(_ficha_t.get("fraccion_63"))
                _cuantia63 = _valor(_pro.get("cuantia")) or _valor(_ficha_t.get("cuantia"))
                _ri63 = _ficha_t.get("resolucion_impugnada") if isinstance(
                    _ficha_t.get("resolucion_impugnada"), dict) else {}
                _aut63 = (_valor(_ri63.get("autoridad")) or _valor(_pro.get("autoridad_demandada"))
                          or _valor(_ficha_t.get("autoridad_demandada")))
            _p_rf, _av_rf = _pf.parrafo(
                _fuente, _anio, fecha_sentencia=_f_sent,
                fecha_interposicion=getattr(computo, "presentacion", None),
                fecha_notificacion=getattr(computo, "notificacion", None),
                fraccion=_fr63, cuantia=_cuantia63 or None, autoridad=_aut63,
                # EL SENTIDO DE LA SALA (3-oct-2026, integración): con él avisa
                # si la nulidad parece formal en la fracción VI.
                sentido=(_valor(((_ficha_t or {}).get("acto") or {}).get("sentido"))
                         if isinstance((_ficha_t or {}).get("acto"), dict) else "") if _nuevo else "")
            # SI EL RECURSO SE DESECHA POR EXTEMPORÁNEO, NO HAY PROCEDENCIA QUE
            # COMPROBAR (3-oct-2026, RF 2, 7 y 6/2025 del banco): el considerando
            # no se escribe y sus avisos pedían revisar —o escribir— un apartado
            # que no existe («Va en hueco: escríbelo»).
            _desecha_rf = bool(
                getattr(computo, "cierra_por_extemporaneidad", None)
                if hasattr(computo, "cierra_por_extemporaneidad")
                else getattr(computo, "oportuna", None) is False)
            if not (_nuevo and _desecha_rf):
                _avisos_bk.extend(_av_rf)
            if _p_rf:
                _proc = _p_rf
        except Exception:
            pass
    # Y EN LOS CUATRO TIPOS, LA DEL BANCO ANTES QUE LA DEL MODELO. Éste era el
    # apartado que tomaba SIEMPRE el texto libre, y por eso la queja decía «El
    # recurso de queja es procedente y no se advierte causa de improcedencia»
    # —una fórmula que no funda nada— teniendo el banco medida la suya, que
    # cita el artículo 97, fracción I, con su inciso y con el auto que se
    # impugna. Es el mismo patrón por tercera vez: una fórmula del corpus que
    # existe y a la que nadie llama.
    if not (_ta.normalizar(tipo_asunto) == "revision_fiscal" and _proc
            != (estructura.procedencia or "")):
        _del_banco_proc = _del_banco(
            "procedencia", "",
            variante=("qj-c2-fraccion-ii" if (_tn == "queja" and _fr97 == "II") else ""))
        if (_del_banco_proc or "").strip() and _fr97_en_hueco:
            _del_banco_proc = _fraccion_97_en_hueco(_del_banco_proc, _huecos_bk, "procedencia")
        if (_del_banco_proc or "").strip():
            _proc = _del_banco_proc
    # ═══ LA PROCEDENCIA DE LA REVISIÓN (C1, 3-oct-2026) [siempre] ══════════
    # Sustituye a la existencia que ya no se reproduce (David: «ya viene en la
    # sentencia recurrida»). Va con y sin la ficha y nunca del modelo: el inciso
    # del 81, fracción I, lo decide lo recurrido (`_clase_ar`, la misma clase
    # que funda la competencia) y, en la suspensión, el inciso que ya escribió
    # la competencia —a) la interlocutoria que resolvió sobre la suspensión
    # definitiva; b) la que modificó o revocó lo resuelto sobre ella—, para que
    # los dos considerandos citen la misma letra.
    if _tn == "amparo_revision":
        _inc_susp = ""
        if _clase_ar == "interlocutoria_suspension":
            _m_inc = re.search(r"(?<!\d)81,\s*fracci[óo]n\s+I,\s*inciso\s+([ab])\)", _comp or "")
            _rx_mod = getattr(_ta, "_RX_MODIFICA_SUSP", None)
            _inc_susp = (_m_inc.group(1) if _m_inc else
                         "b" if (_rx_mod is not None
                                 and _rx_mod.search(str(datos.get("acto") or "")[:1500]))
                         else "a")
        _proc = _ta.procedencia_revision(_clase_ar, _inc_susp)
    # NO SE DECLARA PROCEDENTE LO QUE SE VA A DESECHAR. Cuando el cómputo
    # cierra por extemporaneidad, la ejecutoria tiene un solo resolutivo —«se
    # desecha por extemporáneo»— y este apartado escribía, dos considerandos
    # antes, «El recurso es procedente conforme al artículo 63, fracción VI».
    # Salió firmado así en la revisión fiscal 2/2026: el SEGUNDO concluye la
    # extemporaneidad, el TERCERO declara procedente el recurso y el SEXTO lo
    # desecha. La oportunidad es presupuesto de la procedencia, de modo que
    # pronunciarse sobre ésta después de negar aquélla no es sólo redundante:
    # es contradictorio, y es lo primero que salta al leer el proyecto.
    _cierra_extemp = bool(
        getattr(computo, "cierra_por_extemporaneidad", None)
        if hasattr(computo, "cierra_por_extemporaneidad")
        else getattr(computo, "oportuna", None) is False)
    # Tampoco se declara procedente lo que se va a sobreseer por cumplimiento.
    _cierra_extemp = _cierra_extemp or _sobresee_por_cumplimiento(tipo_asunto)
    # LA ALTERNATIVA ES A LA EXISTENCIA QUE SE ESCRIBIÓ, no a la del modelo
    # (3-oct-2026). Se miraba `estructura.existencia` —el texto libre del
    # modelo—: si venía vacío salía «Procedencia.» con su prosa aunque el banco
    # ya hubiera escrito la existencia, y el documento llevaba los dos
    # considerandos con el ordinal corrido.
    # SIN LA FICHA, LA CONDICIÓN DE SIEMPRE (D5, 3-oct-2026): la regresión sin
    # bandera halló que «Procedencia.» desaparecía cuando el modelo dejaba vacía
    # la existencia; mirar `_exi` es del camino nuevo.
    if (_proc or "").strip() and not _cierra_extemp and (
            esq.get("procedencia_propia")
            or (esq["existencia"] and not ((_exi if _nuevo else estructura.existencia) or "").strip())):
        con_apartados.append(("Procedencia.",
                              (lambda c: lambda p: _texto_en(p, c))(_proc)))

    if _procedencia_primero:
        con_apartados.append(_legit)

    # LA DISPENSA. El rótulo promete el acto reclamado y los conceptos, y el
    # contenido es justamente que NO hace falta transcribirlos: 21 de 26.
    # LA CITA QUE EL MODELO METE DENTRO DE LA DISPENSA se retira: la escribe el
    # compositor debajo, con su rubro en negrita y su registro comprobado. Se
    # corta la frase entera —de «Al respecto/Sirve de apoyo/Sustenta…» hasta el
    # final de la comilla— y no sólo el número, para no dejar un «es aplicable
    # la jurisprudencia» colgando sin decir cuál.
    _RX_CITA_DENTRO = re.compile(
        r"\s*(?:Al\s+respecto,?\s+)?(?:es\s+aplicable|sirve\s+de\s+apoyo|"
        r"sustenta\s+esa\s+consideraci[óo]n|resulta\s+aplicable|"
        r"tiene\s+aplicaci[óo]n)[^.]{0,120}?jurisprudencia[^«“\"]{0,160}"
        r"[«“\"][^»”\"]{20,}[»”\"]\.?\s*", re.I | re.S)

    def _sin_la_cita(t: str) -> str:
        return _RX_CITA_DENTRO.sub(" ", t or "").strip()

    def _dispensa(p):
        _texto_en(
            p,
            _con_su_clase(_sin_la_cita(_del_banco("dispensa", ""))) or
            _contraer(f"Es innecesario transcribir el contenido de "
                      f"{esq['recurrido']} y los {q} hechos valer, pues el deber formal "
                      f"y material de "
                      f"exponer los argumentos legales que sustenten esta resolución no "
                      f"depende de la reproducción literal de los aspectos que conforman "
                      f"la litis, sino de su adecuado análisis."))
        # LA SALVEDAD Y LA TESIS QUE LO SOSTIENE. La dispensa sin su apoyo es
        # una afirmación desnuda, y David la escribió a mano al ajustar el
        # adelanto. Van las dos: la reserva de transcribir cuando el estudio lo
        # pida —que es lo que la hace honesta— y la jurisprudencia que autoriza
        # no transcribir.
        # «DE EL AUTO RECURRIDO» NO ES ESPAÑOL (3-oct-2026, la queja): la
        # preposición y el artículo se juntan aquí, y se contraen.
        parrafo(doc, _con_su_clase(_contraer(
            "No obstante, en el caso de que el estudio demande la transcripción de algún "
            "apartado de " + esq["recurrido"] + f" o de los {q}, así se reflejará.")))
        # LA TESIS, UNA SOLA VEZ Y LA BUENA. El modelo ya la cita —«Al
        # respecto, es aplicable la jurisprudencia 2a./J. 58/2010…»— y yo
        # añadía la mía detrás: el mismo criterio dos veces seguidas con dos
        # redacciones distintas. Puse un párrafo fijo sin mirar si el de arriba
        # ya decía lo mismo.
        #
        # Y NO BASTA CON CALLARME SI ÉL YA LA DIJO, que fue mi primer arreglo:
        # entonces sobrevivía la suya, que no lleva el rubro en negrita ni
        # garantiza el registro. Manda la del compositor, y la del modelo se
        # borra del párrafo de encima.
        _cita_con_rubro(doc, apoyo_dispensa(tipo_asunto))

    # ═══ EL ADHESIVO (3-oct-2026, bandera) ════════════════════════════════
    # El ESQUELETO declaraba «Amparo adhesivo.» y «Revisión adhesiva.» y nada
    # los emitía: en el 722/2025 había amparo adhesivo y el proyecto no tenía
    # ni su legitimación y oportunidad (art. 182 LA) ni su punto resolutivo
    # (8 de 8 versiones). Si la ficha de trámite lo trae, su considerando va
    # antes de la dispensa; lo compone `resultandos_por_tipo`, que es quien
    # conoce las fechas del adhesivo y su cómputo.
    _adh = None
    if _rige_pt and _tn in ("amparo_directo", "amparo_revision", "revision_fiscal"):
        _adh = (_pro.get("adhesivo") if isinstance(_pro.get("adhesivo"), dict) else None) \
            or (_ficha_t.get("adhesivo") if isinstance(_ficha_t.get("adhesivo"), dict) else None)
        if _adh and not any(_valor(v) for v in _adh.values()):
            _adh = None
        # SIN NINGÚN AUTO NO HAY ADHESIVO QUE RESOLVER (3-oct-2026, cuarta ronda,
        # E5; AD 274/2025 del banco: la tercera interesada sólo alegó, la ficha
        # la puso como adherente y el proyecto llevó su considerando con tres
        # huecos y un resolutivo «se declara sin materia el amparo adhesivo»).
        # Si no consta ni la admisión ni la presentación, ni el considerando ni
        # el punto: el resultando lo menciona con hueco y un aviso pregunta.
        # La marca del compositor (`procesal.adhesivo_sin_auto`) es la misma regla.
        if _adh and (_pro.get("adhesivo_sin_auto") is True
                     or not (_valor(_adh.get("admision")) or _valor(_adh.get("presentacion")))):
            _adh = None
            _que_adh = "AMPARO ADHESIVO" if _tn == "amparo_directo" else "REVISIÓN ADHESIVA"
            _ya_adh = [a_ for a_ in (_avisos_de_fuera + list(_avisos_bk))
                       if "¿HUBO" in str(a_) and "ADHESIV" in str(a_)]
            if not _ya_adh:
                _avisos_bk.append(
                    f"¿HUBO {_que_adh}? No consta el auto que lo admite ni la fecha en que se "
                    f"presentó (adhesivo.admision, adhesivo.presentacion): no se escribieron su "
                    f"considerando ni su punto resolutivo. Si lo hubo, escribe en la ficha la fecha "
                    f"del auto que lo admite y vuelve a generar; si no, quita a quien aparece como "
                    f"adherente (adhesivo.quien).")
    if _adh:
        try:
            import resultandos_por_tipo as _rpt
            _rot_a, _txt_a, _av_a = _rpt.considerando_adhesivo(_tn, _ficha_t, datos)
            _avisos_bk.extend(a_ for a_ in (_av_a or []) if a_)
            if str(_txt_a or "").strip():
                con_apartados.append(((str(_rot_a or "").strip().rstrip(".")
                                       or "Legitimación y oportunidad del adhesivo") + ".",
                                      (lambda c: lambda p: _texto_en(p, c))(str(_txt_a))))
        except Exception as _ea:
            _avisos_bk.append(
                f"CONSTA {'AMPARO ADHESIVO' if _tn == 'amparo_directo' else 'REVISIÓN ADHESIVA'} "
                f"Y SU CONSIDERANDO NO SE PUDO COMPONER ({type(_ea).__name__}): escribe su "
                f"legitimación y su oportunidad antes de firmar.")

    # ═══ CONEXIDAD O HECHO NOTORIO (C6, 3-oct-2026) ═══════════════════════
    # Sólo si el secretario marcó asuntos relacionados («con un clic», David).
    # Va después de todo el bloque de legitimación y oportunidad —la
    # procedencia y, si lo hay, el adhesivo— y justo antes de la dispensa,
    # donde lo pone el AD 469/2024 del banco («CUARTO. Conexidad. Con vista en
    # la conexión que guarda…»); el ordinal del estudio se corre solo porque
    # los ordinales se calculan al emitir. El hecho notorio cita el código
    # supletorio de la sede (CFPC 88 o CNPCF 269, `tipos_asunto.supletorio`),
    # el mismo que funda las documentales.
    if _rel_lista:
        try:
            _rot_rel, _txt_rel = _ta.considerando_relacionados(
                tipo_asunto, _rel_num, _rel_mat, _rel_lista,
                str(_sup.get("hecho_notorio") or ""))
        except Exception as _erel:
            _rot_rel, _txt_rel = "", ""
            _avisos_bk.append(
                f"HAY ASUNTOS RELACIONADOS Y SU CONSIDERANDO NO SE PUDO COMPONER "
                f"({type(_erel).__name__}): escribe la conexidad o el hecho notorio antes de firmar.")
        if str(_txt_rel or "").strip():
            if (any(r_.get("estado") == "resuelto" for r_ in _rel_lista)
                    and _sup.get("aviso") and _sup["aviso"] not in _avisos_bk):
                _avisos_bk.append(_sup["aviso"])
            # EL 64 DE LA LFPCA SE AFIRMA, NO SE COMPRUEBA (revisión de normas y
            # front, 3-oct-2026). Rige sólo si el amparo directo reclama la misma
            # sentencia que impugna la revisión fiscal; la tarjeta sólo dice que
            # los dos se relacionan. Cada vez que el considerando lo cita, el
            # secretario lo coteja.
            if "artículo 64 de la Ley Federal de Procedimiento Contencioso" in str(_txt_rel):
                _av_64 = ("EL CONSIDERANDO DE RELACIONADOS CITA EL ARTÍCULO 64 DE LA LFPCA: "
                          "afirma que el amparo directo y la revisión fiscal impugnan "
                          "la misma sentencia y se resuelven en la misma sesión. Compruébalo: sólo rige "
                          "si el amparo directo reclama la misma sentencia que la revisión fiscal; si "
                          "se relacionan por otra causa, quita esa oración y deja la fórmula general.")
                if _av_64 not in _avisos_bk:
                    _avisos_bk.append(_av_64)
            con_apartados.append(((str(_rot_rel or "").strip().rstrip(".")
                                   or "Asuntos relacionados") + ".",
                                  (lambda c: lambda p: _texto_en(p, c))(str(_txt_rel))))

    con_apartados.append((esq["dispensa"].format(q=q), _dispensa))

    if antecedentes:
        def _antecedentes(p):
            # EN PROSA, SIN NÚMEROS (2-oct-2026, David: «debemos quitar la
            # enumeración de antecedentes»; bandera `antecedentes_en_prosa`).
            # Esto REVIERTE la numeración de abajo, y a sabiendas: su razón era
            # que el estudio remite a ellos por su número, pero el estudio nunca
            # los recibe —`fase6_estudio.prompt_estudio` sólo lee el resumen del
            # acto y el de los conceptos— y la moderna ya prohíbe remitir por
            # número, así que no hay remisión escrita por la máquina que romper.
            # El número que traiga el modelo se LIMPIA (regex estrecha: 1-2
            # dígitos con punto o paréntesis al abrir el párrafo; no toca «15 de
            # marzo…» ni los efectos numerados, que van en otro apartado). Y UNA
            # sola fórmula de entrada: el prompt ya no la pide, y si el modelo
            # la escribe sola se borra; si la fundió con el primer hecho, se
            # queda la suya y el documento no añade la propia.
            import contexto_taller as _ct_ant
            if _ct_ant.rediseno("antecedentes_en_prosa"):
                import fases123_resumenes as _fr_ant
                _ps = antecedentes.split("\n") if isinstance(antecedentes, str) else antecedentes
                _ps, _trae_entrada = _fr_ant.antecedentes_en_prosa(_ps)
                if not _trae_entrada:
                    _texto_en(p,
                              "Previo al análisis de los planteamientos que se proponen, "
                              "es menester relatar los hechos relevantes del asunto.")
                for x in _ps:
                    parrafo_con_citas(doc, x, notas)
                return
            _texto_en(p,
                      "Previo al análisis de los planteamientos que se proponen, "
                      "es menester relatar los hechos relevantes del asunto.")
            # NUMERADOS, con la bandera apagada. Así los ajustó David —1 a 19
            # en el adelanto del 410/2026— pensando en que el estudio remitiera
            # a ellos («como se dijo en el antecedente 7»). El número se pone
            # aquí y no se le pide al modelo, que ya lleva bastantes reglas.
            # (El 2-oct-2026 él mismo pidió quitarlo: ver arriba.)
            _n = 0
            for x in (antecedentes or []):
                x = x.strip()
                if not x:
                    continue
                _n += 1
                # Si el modelo ya lo numeró, no se numera dos veces.
                if not re.match(r"^\d{1,2}[.)]\s", x):
                    x = f"{_n}. {x}"
                parrafo_con_citas(doc, x, notas)
        con_apartados.append(("Antecedentes.", _antecedentes))

    # EL ESTUDIO. Es el ÚLTIMO considerando salvo que detrás vaya Efectos, y
    # lleva dentro los subtítulos en negrita, sin ordinal.
    _cuerpo_estudio, _efectos_escritos = partir_efectos(estudio or [])
    _cuerpo_estudio, _cuerpo_conceptos = partir_conceptos(_cuerpo_estudio)
    # LA FRASE QUE REMITE A UNA TRANSCRIPCIÓN QUE NO ESTÁ. Se repara aquí, una
    # vez, sobre los tres cuerpos: el estudio, el de los conceptos y los
    # efectos. La causa se corrigió en el prompt —la arquitectura ordenaba
    # transcribir el precepto mientras el documento lo bajaba al pie—; esto es
    # la red por debajo.
    _cuerpo_estudio = _sin_anuncio_vacio(_cuerpo_estudio)
    _cuerpo_conceptos = _sin_anuncio_vacio(_cuerpo_conceptos)
    _efectos_escritos = _sin_anuncio_vacio(_efectos_escritos)

    # EN UN RECURSO NO SE DEJA NADA INSUBSISTENTE. David: «en revisión la
    # sentencia no se deja insubsistente, se revoca; sólo en amparo (cuando se
    # concede) se ordena que se deje insubsistente el acto reclamado».
    #
    # NO SE CORRIGE EL TEXTO, SE AVISA: la frase puede venir dentro de un
    # razonamiento largo —o describiendo lo que hizo OTRO órgano, que es
    # legítimo— y reescribirla a ciegas es la clase de remiendo que ya salió
    # peor que el defecto. Aquí basta con que quien firma lo vea.
    if _ta.normalizar(tipo_asunto) in ("amparo_revision", "queja",
                                       "revision_fiscal"):
        _txt_rec = " ".join(str(x) for x in
                            (list(_cuerpo_estudio) + list(_cuerpo_conceptos)))
        if re.search(r"\bdej(?:e|ar|ará|en)\s+insubsistente", _txt_rec, re.I):
            _avisos_bk.append(
                "EL ESTUDIO DICE «DEJE INSUBSISTENTE» Y ESTO ES UN RECURSO. En "
                "revisión la sentencia no se deja insubsistente: SE REVOCA, y "
                "con eso deja de existir. Dejar insubsistente es la fórmula del "
                "amparo, donde el tribunal no revoca el acto reclamado sino que "
                "ordena a la responsable retirarlo. Compruébalo: si describe lo "
                "que hizo otro órgano, está bien; si es lo que resuelve este "
                "tribunal, cámbialo por «se revoca».")

    def _cierre_del_estudio():
        if not concede:
            parrafo(doc, _con_su_clase(_ta.parrafo_cierre(tipo_asunto, False)))
        elif not _ta.cierre_de(tipo_asunto)["efectos"]:
            parrafo(doc, _con_su_clase(_ta.parrafo_cierre(tipo_asunto, True,
                                                          _calificacion_plural(cs))))
            _efectos_de_la_revision()

    def _efectos_de_la_revision():
        """EN LA REVISIÓN QUE CONCEDE, LOS EFECTOS CIERRAN EL ÚLTIMO CONSIDERANDO
        (27-sep-2026). `partir_efectos` los sacaba del estudio en los cuatro
        tipos y sólo el amparo directo los volvía a escribir: si el colegiado
        revocaba y concedía —o modificaba los efectos—, el resolutivo remitía a
        «los efectos precisados en el último considerando» y el documento no
        tenía ninguno. Van aquí, bajo su subtítulo, porque el último
        considerando es a donde apuntan los puntos resolutivos de la revisión;
        un considerando propio correría el ordinal que el corpus no tiene."""
        if _ta.normalizar(tipo_asunto) != "amparo_revision" or not _efectos_escritos:
            return
        import fase_rama as _fr_ef
        _txt_ef = " ".join(str(x) for x in (estudio or []))
        if not (_fr_ef.sentido_en_plenitud(_txt_ef) == "concede"
                or _fr_ef.solo_los_efectos(_txt_ef)):
            return
        _ords, _avs = componer_efectos(_efectos_escritos)
        _avisos_bk.extend(_avs)
        _subtitulo(doc, "Efectos de la concesión")
        if _ords:
            parrafo(doc, _ta.APERTURA_EFECTOS)
            for _o in _ords:
                if sin_andamio(_o).strip():
                    parrafo(doc, sin_andamio(_o).strip())
        else:
            for _x in _efectos_escritos:
                if sin_andamio(_x).strip():
                    parrafo(doc, sin_andamio(_x).strip())

    def _estudio(p):
        calif = _calificacion_plural(cs)
        # «LOS AGRAVIOS SON SIN MATERIA» NO ES ESPAÑOL. Con las cinco
        # calificaciones que se conjugan con «ser» la fórmula vale; «sin
        # materia» no se es, se QUEDA. Un secretario escribe «los agravios
        # quedaron sin materia», y el encabezado del estudio es de las frases
        # que se leen antes que ninguna otra.
        if calif and calif.startswith("sin materia"):
            _texto_en(p, f"Los {q} quedaron sin materia.")
        elif calif and "sin materia" in calif:
            # Mezcla: «en parte inoperantes y en parte sin materia» → se
            # reordena para que el verbo case con las dos.
            _texto_en(p, f"Los {q} son, en parte, "
                         f"{calif.replace('en parte ', '').replace(' y en parte ', ' y, en parte, quedaron ')}.")
        else:
            _texto_en(p, f"Los {q} son {calif}." if calif else "")
        if resumen_acto:
            _subtitulo(doc, esq["sub_recurrido"])
            for x in resumen_acto:
                if x.strip():
                    parrafo_con_citas(doc, x.strip(), notas)
        if resumen_conceptos:
            _subtitulo(doc, q[0].upper() + q[1:])
            for x in resumen_conceptos:
                if x.strip():
                    parrafo_con_citas(doc, x.strip(), notas)
        if (marco_escrito or "").strip():
            _subtitulo(doc, "Marco jurídico")
            for x in re.split(r"\n\s*\n", marco_escrito):
                if x.strip():
                    parrafo(doc, x.strip())
        _subtitulo(doc, "Solución")
        # EL CIERRE, UNA SOLA VEZ. Debajo de esto el documento añade la fórmula
        # del tipo —«lo procedente es confirmar la sentencia recurrida»—, y el
        # modelo escribe la suya por iniciativa propia: el proyecto acababa con
        # dos cierres seguidos diciendo lo mismo.
        #
        # Se le dice en el prompt Y se quita aquí, porque una instrucción al
        # final de un prompt de veintidós mil caracteres se pierde —ya van
        # varias— y esto tiene que salir bien siempre. Se recorta SÓLO la
        # frase de remate, no el párrafo: la recapitulación del modelo dice qué
        # agravio es infundado y por qué, y eso es sustancia que no sobra.
        # NOMBRE NUEVO, NO REASIGNACIÓN. `_cuerpo_estudio` viene del ámbito de
        # fuera; asignarlo aquí lo vuelve local de esta función y la lectura de
        # la derecha revienta con UnboundLocalError. Es el mismo tropiezo de
        # ámbito que ya costó una ronda en kaisen3-fase1, y esta vez lo cazó el
        # guardián antes de salir.
        _cuerpo_sin_remate = _sin_remate_duplicado(_cuerpo_estudio)
        _escribir_estudio(doc, _cuerpo_sin_remate, tesis, notas, normas)
        # EL CIERRE ES DEL TIPO. Aquí decía «lo procedente es negar el amparo
        # solicitado» en los cuatro, incluida la queja, que además decretaba
        # «Es infundado el recurso de queja» treinta líneas más abajo: el mismo
        # documento afirmaba dos desenlaces incompatibles.
        #
        # Y CUANDO CONCEDE, EL CIERRE VA AQUÍ TAMBIÉN salvo en el amparo
        # directo, que lleva su apartado de «Efectos». Ponerlo en un apartado
        # propio para los recursos correría el ordinal y dejaría el Estudio
        # donde el corpus no lo tiene.
        # EL CIERRE VA DONDE TERMINA DE RESOLVERSE EL ASUNTO. Si detrás hay
        # un considerando de conceptos de violación, escribirlo aquí anuncia
        # el desenlace ANTES de estudiar aquello de lo que depende: el
        # documento diría «procede conceder el amparo» y acto seguido se
        # pondría a examinar si los conceptos son fundados. Es la misma
        # incongruencia del resolutivo que negaba lo que el estudio concedía,
        # entrando por otra puerta.
        if not _cuerpo_conceptos:
            _cierre_del_estudio()

    # ═══════════════════════════════════════════════════════════════════════
    # SI EL CÓMPUTO DA EXTEMPORÁNEA, NO HAY FONDO
    # ═══════════════════════════════════════════════════════════════════════
    # David: «si oportunidad == Extemporánea, el flujo debe abortar el estudio
    # de fondo y generar automáticamente el sobreseimiento». La falla que
    # describe —declarar la extemporaneidad y luego conceder— no se corrige
    # avisando: se corrige no escribiendo el estudio.
    #
    # El apartado de fondo se sustituye por el de improcedencia, que es lo que
    # el corpus escribe: «{ordinal}. Extemporaneidad del recurso de revisión.
    # El presente medio de impugnación se interpuso de manera extemporánea».
    # LA CONDICIÓN SE LE PREGUNTA AL CÓMPUTO, NO SE REARMA AQUÍ. Gobierna tres
    # decisiones de este documento —el apartado de fondo, el de efectos y el
    # punto resolutivo— y estaba escrita a mano en la primera y copiada en las
    # otras dos. Con la decisión del secretario de por medio serían tres copias
    # de una condición de cuatro términos, y basta que una se quede sin
    # actualizar para que salga un proyecto con estudio de fondo y resolutivo
    # de sobreseimiento: la misma incongruencia que esto existe para impedir,
    # entrando por la puerta de enfrente. El `getattr` es para un Computo
    # viejo que llegue sin las propiedades nuevas.
    _extemp = (getattr(computo, "cierra_por_extemporaneidad", None)
               if hasattr(computo, "cierra_por_extemporaneidad")
               else (getattr(computo, "oportuna", None) is False
                     and not getattr(computo, "en_cualquier_tiempo", False)))
    _extemp = bool(_extemp)
    # EL GEMELO: SOBRESEIMIENTO POR CUMPLIMIENTO (30-sep-2026). La
    # extemporaneidad manda si concurren (es presupuesto de todo lo demás).
    _cumpl_sob = (not _extemp) and _sobresee_por_cumplimiento(tipo_asunto)
    _reserva = bool(getattr(computo, "fondo_en_reserva", False))
    _rectif = bool(getattr(computo, "rectificada", False))
    if _rectif:
        _avisos_bk.insert(0, (
            f"EL CÓMPUTO DA EXTEMPORÁNEA Y TÚ DECLARASTE QUE NO. El proyecto "
            f"entra al fondo, y el considerando de oportunidad lleva el "
            f"desglose completo del cómputo seguido de tu razón, literal: "
            f"«{(getattr(computo, 'motivo', '') or '')[:240]}». LÉELA EN EL "
            f"PAPEL antes de firmar: es lo único que sostiene la competencia "
            f"para estudiar el fondo, y si no se sostiene el sobreseimiento "
            f"vuelve en revisión."))
    if _extemp:
        _ex = _ta.extemporaneo_de(tipo_asunto)
        import fase0_oportunidad as _f0a
        _conf_av = _f0a.conforme_a(_ex["fundamento"])
        if _reserva:
            _avisos_bk.insert(0, (
                f"EL CÓMPUTO DA EXTEMPORÁNEA Y PEDISTE EL ESTUDIO EN RESERVA. "
                f"La ejecutoria resuelve la improcedencia conforme "
                f"{_conf_av}, con su punto resolutivo. El estudio de "
                f"fondo va DETRÁS de los resolutivos, en un anexo rotulado que "
                f"no forma parte de la ejecutoria y que no se notifica. Si el "
                f"Pleno no comparte la extemporaneidad, ese anexo es el "
                f"engrose; si la comparte, bórralo antes de listar."))
        else:
            _avisos_bk.insert(0, (
                f"EL CÓMPUTO DA EXTEMPORÁNEA: el proyecto NO entra al fondo y "
                f"resuelve la improcedencia conforme {_conf_av}. Si "
                f"la fecha de notificación o la de presentación están mal, "
                f"corrígelas y vuelve a generar: de esas dos fechas depende "
                f"todo el asunto. Y SI EL CÓMPUTO ESTÁ INCOMPLETO —porque la "
                f"autoridad responsable suspendió labores o tuvo periodo "
                f"vacacional en días que el calendario federal tiene por "
                f"hábiles— dilo tú: en la pantalla de resolución, «la "
                f"oportunidad la decido yo», escribiendo la razón. El proyecto "
                f"entrará al fondo con esa razón en el considerando."))
        con_apartados.append(
            (_ex["rotulo"] + ".",
             (lambda c: lambda p: _texto_en(p, c))(_ex["considerando"])))
    elif _cumpl_sob:
        import cumplimiento_ejecutoria as _ce_d
        _avisos_bk.insert(0, (
            "CONFIRMASTE QUE LA EJECUTORIA NO DEJÓ LIBERTAD DE JURISDICCIÓN: el proyecto NO entra al fondo y "
            "sobresee (artículos 61, fracción IX, y 63, fracción V, de la Ley de Amparo; 2a./J. 113/2012 y "
            "1a./J. 57/2018). Si alguna parte se resolvió con libertad, quita la confirmación en «De dónde viene "
            "lo reclamado» y vuelve a generar: con libertad parcial no se sobresee. LA VISTA DEL ARTÍCULO 64, "
            "PÁRRAFO SEGUNDO: si la causa se advierte de oficio, precisa en el considerando cuándo se dio vista a la "
            "quejosa y qué contestó (o razona por qué no hacía falta); si la hizo valer una parte, quita ese párrafo."))
        con_apartados.append(
            (_ce_d.IMPROCEDENTE["rotulo"] + ".",
             (lambda c: lambda p: _texto_en(p, c))(_ce_d.considerando_improcedencia())))
    else:
        # ── LA CUESTIÓN, FIJADA ANTES DE RESOLVERLA ────────────────────────
        # Roberto Lara Chagoyán, «Sobre la estructura de las sentencias en
        # México», § 3.2 y § 4.2: «la desgracia de muchas malas sentencias
        # comienza con el descuido del deber de fijar cuidadosamente la
        # cuestión… Una forma de mejorar los planteamientos es utilizar la
        # PREGUNTA EXPRESA», y su ejemplo de apartado: «TERCERO. Materia de la
        # revisión. Se constriñe a determinar si la parte quejosa logra, con
        # sus agravios, desvirtuar las razones por las que el Juez de Distrito
        # negó el amparo».
        #
        # POR QUÉ SE COMPONE Y NO SE PIDE. Se lo pedí al modelo en el prompt y
        # no lo hizo: medido, 5 de 6 problemas sin pregunta, y después del
        # EL CONSIDERANDO DE MATERIA SE RETIRA, y con él una idea mía que
        # resultó equivocada.
        #
        # David: «parece innecesario el considerando adicional de materia del
        # recurso… no es necesario bajar esos problemas jurídicos en un
        # considerando aparte. Es suficiente con hacer referencia al agravio o
        # al concepto de violación para enseguida calificarlo. El tema de los
        # problemas jurídicos sí es útil, pero para GUIAR EL ESTUDIO, no
        # propiamente para plasmarlo en el proyecto».
        #
        # Tiene razón, y la confusión era mía: las preguntas son el andamio con
        # el que se construye el razonamiento, no parte de lo construido. Un
        # apartado que las enumera obliga al lector a leerlas dos veces —una en
        # la lista y otra al contestarlas— y no aporta nada que el estudio no
        # diga mejor.
        #
        # NO SE PIERDE NADA DEL TRABAJO: las preguntas se siguen calculando, se
        # siguen enseñando en pantalla, y siguen entrando en el prompt del
        # estudio ordenadas por prelación lógica. Lo único que cambia es que no
        # se imprimen.

        con_apartados.append((_ta.rotulo_estudio_de(tipo_asunto).rstrip(".") + ".",
                              _estudio))

        # EL CONSIDERANDO DE LOS CONCEPTOS DE VIOLACIÓN, CON SU ORDINAL.
        if _cuerpo_conceptos:
            def _conceptos_ap(p):
                _texto_en(p, _cuerpo_conceptos[0])
                # EL REMATE DEL MODELO SE PODA AQUÍ TAMBIÉN. El apartado de
                # los agravios ya pasaba por esta poda; el de los conceptos
                # nació sin ella y el documento cerraba dos veces —«lo
                # procedente es conceder el amparo» y, debajo, la fórmula
                # compuesta diciendo lo mismo—. Es el mismo defecto que se
                # arregló arriba, entrando por el apartado nuevo.
                _escribir_estudio(doc, _sin_remate_duplicado(_cuerpo_conceptos[1:]),
                                  tesis, notas, normas)
                _cierre_del_estudio()
            con_apartados.append(("Estudio de los conceptos de violación.",
                                  _conceptos_ap))

    # EL APARTADO DE EFECTOS ES DEL AMPARO. «Procede conceder el amparo y
    # protección de la Justicia Federal para el efecto de que la responsable
    # deje insubsistente la sentencia reclamada» no se escribe en una queja
    # fundada: ahí se revoca el auto y se ordena proveer de nuevo. Se emitía en
    # los cuatro tipos, así que era la segunda puerta —la de la rama FUNDADA—
    # por la que la fórmula del amparo entraba en un recurso. Arreglar sólo la
    # otra habría tapado la mitad.
    if _extemp or _cumpl_sob:
        pass          # no hay efectos de una concesión que no existe
    elif concede and _ta.cierre_de(tipo_asunto)["efectos"]:
        # «EFECTOS. CON FUNDAMENTO EN EL ARTÍCULO 77…, LA AUTORIDAD RESPONSABLE
        # DEBERÁ:» y debajo las órdenes (David, 27-sep-2026). La apertura
        # continúa el rótulo; cada orden va en su párrafo.
        _ordenes_ef, _avisos_ef = componer_efectos(_efectos_escritos)
        _avisos_bk.extend(_avisos_ef)
        if not _efectos_escritos:
            # SIN EFECTOS ESCRITOS, EL ESQUELETO CON SU HUECO. La fórmula de
            # antes —«dicte otra en la que atienda los lineamientos de esta
            # ejecutoria»— no se puede ejecutar sin interpretarla, y el banco lo
            # dice sin rodeos: un párrafo de efectos genérico es peor que el
            # esqueleto vacío, porque el mandato del caso es justo donde va el
            # criterio del secretario.
            # LO RECLAMADO POR SU NOMBRE (3-oct-2026, quinta ronda; AD 349/2025
            # del banco: «1. Dejar insubsistente la sentencia reclamada.» con el
            # V I S T O, la existencia y la legitimación diciendo «la resolución
            # reclamada»; con un laudo, igual). Es lo que el compositor expuso
            # (`procesal.acto_reclamado`) o, si no, la clase que usa el resto
            # del documento; la segunda orden concuerda con él.
            _ar_ef = "la sentencia reclamada"
            if _nuevo and _tn == "amparo_directo":
                _ar_p = " ".join(_valor(_pro.get("acto_reclamado")).lower().split())
                if re.fullmatch(r"(?:la\s+(?:sentencia|resoluci[óo]n)\s+reclamada|"
                                r"el\s+laudo\s+reclamado)", _ar_p):
                    _ar_ef = _ar_p.replace("resolucion", "resolución")
                elif _clase_ad in ("resolucion", "laudo"):
                    _ar_ef = ("la resolución reclamada" if _clase_ad == "resolucion"
                              else "el laudo reclamado")
            _otra_ef = ("Emitir otro en el que" if _ar_ef.startswith("el ")
                        else "Emitir una nueva en la que")
            _ordenes_ef = [
                f"1. Dejar insubsistente {_ar_ef}.",
                f"2. {_otra_ef} reitere lo que no fue materia de "
                f"la concesión y {HUECO}.",
                "3. Hecho lo anterior, resolver con plenitud de jurisdicción lo "
                "que en derecho corresponda."]
            _avisos_bk.append(
                "LOS EFECTOS VAN CON UN HUECO: el estudio no los escribió. La "
                "segunda orden debe decir qué tiene que decidir la responsable al "
                "volver a resolver; complétala antes de firmar.")
        elif not _ordenes_ef:
            _avisos_bk.append(
                "LOS EFECTOS VINIERON EN PROSA y no como órdenes: redáctalos "
                "como «Con fundamento en el artículo 77 de la Ley de Amparo, la "
                "autoridad responsable deberá:» y de tres a cinco órdenes "
                "numeradas en infinitivo.")

        def _efectos(p):
            if _ordenes_ef:
                _texto_en(p, _ta.APERTURA_EFECTOS, _ordenes_ef)
            else:
                # Vinieron en prosa, sin una sola orden reconocible: se escriben
                # como vinieron y se avisa, que reescribirlos a ciegas es peor.
                _texto_en(p, _efectos_escritos[0], _efectos_escritos[1:])
        con_apartados.append(("Efectos.", _efectos))

    _emitir(con_apartados)

    # LOS HUECOS DEL BANCO, CON EL NOMBRE DEL DATO (3-oct-2026). Uno por
    # apartado; los que ya tienen su aviso propio (el inciso del 97 y su cola)
    # no se repiten.
    # (la fracción del 63 en la revisión fiscal la avisa `fase_procedencia_rf`).
    _ya_avisados = ({"inciso", "cola_97"} if _tn == "queja" else
                    {"fraccion_63", "motivo_procedencia"} if _tn == "revision_fiscal" else set())
    _por_apartado: dict = {}
    _por_dato: dict = {}
    for _ap, _m in _huecos_bk:
        if _m in _ya_avisados:
            continue
        _por_apartado.setdefault(_ap, [])
        _nom = _NOMBRE_DEL_DATO.get(_m, _m)
        if _nom not in _por_apartado[_ap]:
            _por_apartado[_ap].append(_nom)
        _por_dato.setdefault((_m, _nom), [])
        if _ap not in _por_dato[(_m, _nom)]:
            _por_dato[(_m, _nom)].append(_ap)
    if not _nuevo:
        for _ap, _noms in _por_apartado.items():
            _avisos_bk.append(
                f"EL CONSIDERANDO DE {_ap.upper()} LLEVA HUECO (*********): falta "
                + "; ".join(_noms) + ". Complétalo antes de firmar"
                + (" —en «Trámite en este tribunal», si es un dato del trámite—."
                   if _pro else "."))
    else:
        # CON LA FICHA, UN AVISO POR DATO Y NINGUNO SI OTRA PIEZA YA LO DIO
        # (3-oct-2026, tercera ronda): el compositor avisa del campo que falta
        # («FALTA LA FECHA DEL AUTO RECURRIDO (acto.fecha): está en…»), y es el
        # mismo campo el que llena el hueco del considerando.
        _esp_bk = _avisos_de_fuera + list(_avisos_bk) + list(avisos_doc)
        for (_m, _nom), _aps in _por_dato.items():
            if _clave_ya_avisada(_CLAVES_DEL_MARCADOR.get(_m, ()), _esp_bk):
                continue
            _cab_h = (f"EL CONSIDERANDO DE {_aps[0].upper()} LLEVA" if len(_aps) == 1 else
                      "LOS CONSIDERANDOS DE " + ", DE ".join(a_.upper() for a_ in _aps[:-1])
                      + f" Y DE {_aps[-1].upper()} LLEVAN")
            _avisos_bk.append(
                f"{_cab_h} HUECO (*********): falta {_nom}. Complétalo antes de firmar "
                f"—en «Trámite en este tribunal», si es un dato del trámite—.")

    # ── EL PUNTO DEL ADHESIVO, antes de escribir los demás: si lo hay, el
    # punto principal deja de ser «ÚNICO» (3-oct-2026, bandera) ──
    _adh_res = ""
    if _adh:
        try:
            import resultandos_por_tipo as _rpt_r
            # SIN CALIFICACIÓN NO SE SABE si prospera el principal (el adelanto
            # se compone antes del estudio): None, y el punto va en hueco.
            _prospera = ("desecha" if _extemp else "sobresee" if _cumpl_sob
                         else (bool(concede) if cs else None))
            _t_ad, _av_ad = _rpt_r.resolutivo_adhesivo(_tn, _prospera, _ficha_t)
            _adh_res = _RX_ORDINAL_AL_FRENTE.sub("", str(_t_ad or "")).strip()
            if _av_ad:
                _avisos_bk.append(str(_av_ad))
        except Exception as _ear:
            _avisos_bk.append(
                f"CONSTA UN ADHESIVO Y SU PUNTO RESOLUTIVO NO SE PUDO COMPONER "
                f"({type(_ear).__name__}): añádelo antes de firmar.")
    _n_puntos = [0]

    def _cab_de(cab: str) -> str:
        """«ÚNICO» deja de serlo si detrás va el punto del adhesivo."""
        _n_puntos[0] += 1
        if _adh_res and cab.strip().upper() in ("ÚNICO", "UNICO"):
            return "PRIMERO"
        return cab

    # ── RESUELVE ──
    p_pe = parrafo(doc, "Por lo expuesto y fundado, se:", sangria=True)
    p_pe.paragraph_format.space_before = Pt(14)
    rotulo(doc, "Resuelve")
    _res = RESOLUTIVO.get(_ta.normalizar(tipo_asunto),
                          RESOLUTIVO["amparo_directo"])

    # ═══════════════════════════════════════════════════════════════════════
    # EL AMPARO EN REVISIÓN TIENE DOS PUNTOS, NO UNO
    # ═══════════════════════════════════════════════════════════════════════
    # Está medido en el corpus del propio tribunal, con esas palabras: «en
    # revisión el resolutivo tiene DOS puntos: PRIMERO decide sobre la
    # sentencia recurrida —confirma, modifica, revoca— y SEGUNDO reproduce el
    # sentido del amparo —ampara, no ampara, sobresee—. Sólo hay ÚNICO cuando
    # se desecha el recurso». Y estaba en `banco_plantillas.json` sin que nadie
    # lo leyera: el proyecto salía siempre con «ÚNICO. Se confirma la sentencia
    # recurrida», de modo que un amparo en revisión NUNCA amparaba.
    _hecho = False
    if _extemp:
        _ex = _ta.extemporaneo_de(tipo_asunto)
        _t = _ex["resolutivo"].replace("{quejoso}", _q_prosa or HUECO)
        _cab, _resto = _t.split(". ", 1) if ". " in _t else (_t, "")
        tramos(doc, [(_cab_de(_cab) + ". ", {"bold": True}), (_resto, {})], sangria=False)
        _hecho = True
    elif _cumpl_sob:
        import cumplimiento_ejecutoria as _ce_r
        _t = _ce_r.IMPROCEDENTE["resolutivo"].replace("{quejoso}", _q_prosa or HUECO)
        _cab, _resto = _t.split(". ", 1) if ". " in _t else (_t, "")
        tramos(doc, [(_cab_de(_cab) + ". ", {"bold": True}), (_resto, {})], sangria=False)
        _hecho = True
    elif _ta.normalizar(tipo_asunto) == "amparo_revision":
        # EL ESTUDIO ENTERO, NO SUS PRIMEROS SEIS MIL CARACTERES. `resolvio_a_quo`
        # es un barrido de expresiones regulares: cuesta lo mismo mirarlo todo,
        # y truncarlo sólo puede perder la frase que dice qué resolvió el
        # juzgado. Un resolutivo en hueco por no haber leído el párrafo 40 es
        # un precio absurdo por un ahorro que nadie iba a notar.
        _fuente_rama = " ".join([
            str(datos.get("antecedentes") or ""),
            str(datos.get("acto") or ""),
            " ".join(str(r.get("texto") or "")
                     for r in (estructura.resultandos or [])),
            str(estudio or "")])
        try:
            import fase_rama as _fr
            # LOS ANTECEDENTES MANDAN SOBRE EL ESTUDIO. En el estudio la
            # palabra «sobreseimiento» aparece dentro de las TESIS
            # TRANSCRITAS, y con eso el proyecto confirmaba un sobreseimiento
            # que nadie decretó —medido sobre el ARA 17/2025: con el estudio
            # entero da «sobresee», sin las tesis da «concede»—. Lo que hizo el
            # juzgado está en los antecedentes y en los resultandos, que es
            # donde el catálogo manda escribirlo.
            _antes_rama = " ".join([
                str(datos.get("antecedentes") or ""),
                " ".join(str(r.get("texto") or "")
                         for r in (estructura.resultandos or []))]).strip()
            # LO QUE EL MOTOR YA DIJO AL PREPARAR LA PROPUESTA. Es su
            # respuesta a esta misma pregunta —«Sobreseyó con fundamento en el
            # artículo 63, fracción IV»— dada tras leer el expediente entero.
            # En la revisión 410/2026 los antecedentes narraban el juicio de
            # nulidad y nunca decían en qué paró el amparo, así que el
            # resolutivo salió en hueco mientras el estudio, tres párrafos
            # antes, decía «se confirma» y «debe mantener el sobreseimiento».
            # ═══════════════════════════════════════════════════════════
            # MANDA EL PAPEL, NO LA PROSA
            # ═══════════════════════════════════════════════════════════
            # `resolvio_a_quo` viene leído del PDF de la sentencia recurrida en
            # el adelanto, con un barrido determinista sobre su propio punto
            # resolutivo. Lo de abajo —el barrido sobre los resúmenes del
            # modelo y la frase con la que describió el asunto— sigue como
            # repliegue para las sesiones viejas y para cuando el papel no se
            # dejó leer.
            #
            # De este dato depende el resolutivo entero, y decidirlo con prosa
            # es echarlo a suertes: en la revisión 650/2025 el proyecto
            # confirmó la sentencia y NEGÓ el amparo que esa misma sentencia
            # había concedido, mientras el resultando decía «no consta el
            # sentido de la sentencia recurrida» y el PDF decía, dos páginas
            # antes, «La Justicia de la Unión ampara y protege».
            _que_hizo = str(datos.get("resolvio_a_quo") or "").strip().lower()
            # EL PUNTO RESOLUTIVO DEL JUZGADO MANDA SOBRE EL RECUENTO
            # (28-sep-2026, AR 631/2025). La sesión guardó «niega» —el recuento
            # de verbos del PDF entero, arrastrado por un amparo ANTERIOR que
            # la sentencia narraba— y su propio resolutivo, leído en la misma
            # pasada, decía «ampara y protege». Con «niega» y el recurso
            # fundado, la rama revocó la concesión para volver a conceder.
            _por_puntos = _fr.que_dice_el_resolutivo(
                str(datos.get("resolutivo_recurrida") or ""))
            if _por_puntos and _por_puntos != _que_hizo:
                if _que_hizo:
                    _avisos_bk.append(
                        f"LO QUE HIZO EL JUZGADO SE TOMÓ DE SU PUNTO RESOLUTIVO "
                        f"(«{_por_puntos.replace('_', ' y ')}»), no de la lectura "
                        f"guardada del resto de la sentencia («{_que_hizo.replace('_', ' y ')}»), "
                        f"que contaba verbos de otros juicios. Compruébalo contra el "
                        f"resolutivo del juzgado: de esto depende que se confirme o se "
                        f"revoque, y si se ampara o se niega.")
                _que_hizo = _por_puntos
            # LA SENTENCIA MIXTA, LEÍDA DESPUÉS. Las sesiones de antes de la
            # lectura mixta guardaron «sobresee» a secas aunque el juzgado
            # también negara o concediera por los demás actos (revisión
            # 322/2025). Si los antecedentes o lo declarado dicen las dos
            # cosas con sus verbos, manda eso: es el mismo dato, mejor leído.
            if _que_hizo == "sobresee":
                _que_hizo = (_fr._mixto(_antes_rama)
                             or _fr._mixto(str(datos.get("resolvio_declarado") or ""))
                             or _que_hizo)
            if _que_hizo not in ("sobresee", "niega", "concede",
                                 "sobresee_niega", "sobresee_concede"):
                _que_hizo = _fr.resolvio_a_quo(
                    _fuente_rama, _antes_rama,
                    declarado=str(datos.get("resolvio_declarado") or ""))
                # QUE EL REPLIEGUE SE VEA. Sin este aviso, leer el papel y NO
                # leerlo producen documentos indistinguibles, y un `getattr`
                # con valor por omisión que apunta al objeto equivocado pasa
                # inadvertido —pasó, y el resolutivo volvió a salir de la
                # prosa—. Si el dato determinista falta, que conste.
                _avisos_bk.append(
                    "EL SENTIDO DE LA SENTENCIA RECURRIDA NO SE PUDO LEER DEL "
                    "PDF y se dedujo del texto del proyecto. De ese dato "
                    "depende que se confirme o se revoque: compruébalo contra "
                    "el resolutivo del juzgado antes de firmar.")
            # EL SENTIDO EN PLENITUD SE LEE DEL ESTUDIO, no del recurso. Que el
            # agravio sea fundado prueba que el juez no debió sobreseer, no que
            # el quejoso tenga razón en el fondo.
            _sent_amparo = _fr.sentido_en_plenitud(str(estudio or ""))
            _txt_est = (" ".join(str(x) for x in estudio)
                        if isinstance(estudio, (list, tuple)) else str(estudio or ""))
            _solo_ef = _fr.solo_los_efectos(str(estudio or ""))
            _vp_est = _fr.hay_violacion_procesal(str(estudio or ""))
            # ¿SE REASUME JURISDICCIÓN AL REVOCAR UNA CONCESIÓN? (art. 93, fr.
            # VI; AR 631/2025, 28-sep-2026). Lo que el redactor supo al
            # resolver viaja en `reasuncion`: quién recurre, si los conceptos
            # no estudiados constaron y si la recurrida también sobreseyó. Si
            # no constaron, el estudio no pudo concluir y el punto del amparo
            # NO se afirma: va con hueco, aunque la prosa diga algo.
            _reas_d = datos.get("reasuncion") if isinstance(datos.get("reasuncion"), dict) else {}
            # QUIÉN RECURRE Y SI PROSPERA LA PROCEDENCIA (revisión del 28-sep-2026,
            # AR 631/2025): la quejosa que gana su recurso contra una concesión
            # no pierde el amparo (fr. V) y la improcedencia que prospera
            # sobresee sin reasumir (fr. II). Mismos datos que la rama del
            # estudio (`redactor_adelanto._rama_de`).
            _quien_d = str(_reas_d.get("quien_recurre") or datos.get("papel_recurrente") or "")
            _proc_d = bool(_reas_d.get("procedencia"))
            # SIN PRUEBA DE QUIÉN RECURRE, SE DICE (revisión del 28-sep-2026):
            # `papel_del_recurrente` ya no contesta «quejoso» a ciegas.
            if "quien_recurre" in _reas_d and not _quien_d.strip():
                _avisos_bk.append(
                    "NO CONSTA QUIÉN RECURRE —la quejosa, la tercera interesada o la "
                    "autoridad—: el resolutivo se calculó como si no fuera la quejosa. "
                    "Escribe quién recurre en el encargo y vuelve a generar.")
            _tipo_reas = _ta.reasuncion(
                _que_hizo, "fundado" if concede else "infundado",
                solo_efectos=_solo_ef, violacion_procesal=_vp_est,
                quien_recurre=_quien_d, procedencia=_proc_d)
            # SÓLO SI HACEN FALTA DE VERDAD: «por_confirmar» (la recurrida no dice
            # que quedaran conceptos sin estudiar) no deja el amparo en hueco;
            # manda lo que concluya el estudio.
            if _tipo_reas == "concesion" and _reas_d.get("hacen_falta") is True \
                    and _reas_d.get("tenemos") is False:
                _sent_amparo = ""
            _clave = _ta.rama_revision(
                _que_hizo,
                "fundado" if concede else "infundado",
                solo_efectos=_solo_ef,
                violacion_procesal=_vp_est,
                sentido_amparo=_sent_amparo,
                quien_recurre=_quien_d, procedencia=_proc_d)
            _rama = _ta.RAMAS_REVISION[_clave]
            # ═══════════════════════════════════════════════════════════════
            # EL RESPALDO AMPARABA CONTRA EL ÓRGANO RECURRIDO
            # ═══════════════════════════════════════════════════════════════
            # `datos["responsable"]` en un recurso puede ser el ÓRGANO
            # RECURRIDO —«EL JUZGADO SEGUNDO DE DISTRITO»; la carátula de la
            # revisión ya no lo rotula (0 de 478 en el corpus, AR 631/2025)—,
            # no la responsable originaria. Cuando
            # `responsable_originaria` no lograba leerla del resumen, el
            # respaldo escribía «La Justicia de la Unión ampara y protege a
            # Juan Pérez, contra el acto reclamado AL JUZGADO SEGUNDO DE
            # DISTRITO». Es un disparate: se ampara contra el acto de la
            # autoridad que emitió el acto reclamado del amparo indirecto, no
            # contra el juez que resolvió ese amparo.
            #
            # Este módulo ya tenía la doctrina escrita en `fase_rama`: «Y
            # CUANDO NO CONSTA, SE DEJA EL HUECO». Se aplica. El comodín se ve,
            # el linter lo cuenta y el aviso dice dónde buscarla.
            #
            # Y PASA POR `_con_articulo`, que era el otro defecto: el nombre
            # sale del resumen sin artículo y el punto resolutivo decía «contra
            # el acto reclamado a Director de Ingresos». El artículo se pone
            # aquí y `_contraer` hace el resto —«a el» → «al»—.
            _orig = _con_articulo(_fr.responsable_originaria(_antes_rama)
                                  or _fr.responsable_originaria(_fuente_rama))
            _aviso_todas = ""
            # CUANDO RECURRE LA AUTORIDAD, EL ACTO ES EL SUYO (23-sep-2026).
            # En el 711/2025 el amparo señalaba a varias responsables y el
            # juzgado sobreseyó respecto de casi todas; la lectura del texto
            # tomó la primera —«la Titular de la Secretaría de Hacienda»— y el
            # resolutivo negó el amparo contra el acto de quien no dictó nada.
            # Si quien recurre es una autoridad, recurre porque el acto
            # reclamado es suyo y se lo concedieron en contra: ésa es la
            # responsable originaria, sin leer nada.
            _recurrente_res = str(datos.get("recurrente") or "").strip()
            if _nuevo:
                # Sin la etiqueta de rol ni las versales de la carátula: «contra
                # el acto reclamado al GOBERNADOR … (AUTORIDAD RESPONSABLE)».
                _recurrente_res = _en_prosa(_recurrente_res)
            if _recurrente_res and _rec_es_autoridad(_recurrente_res):
                _orig = _con_articulo(_recurrente_res)
            # LA FICHA SABE QUIÉN FUE LA RESPONSABLE (3-oct-2026, bandera): las
            # autoridades de la demanda de amparo indirecto, ancladas al papel
            # (`datos_extra["autoridades"]`), o la responsable del acto. Antes de
            # dejar el punto con comodín se toman de ahí; si son varias, todas,
            # cada una con su artículo.
            if not _orig and _rige_pt and _pro:
                _auts = [a_ for a_ in (_pro.get("autoridades") or [])
                         if isinstance(a_, str) and _valor(a_)]
                if not _auts and _valor(_pro.get("responsable")):
                    _auts = [_valor(_pro.get("responsable"))]
                _auts = [_con_articulo(_valor(a_)) for a_ in _auts]
                _auts = [a_ for a_ in _auts if a_]
                if _auts:
                    _orig = (_auts[0] if len(_auts) == 1
                             else ", ".join(_auts[:-1]) + " y " + _auts[-1])
                    if len(_auts) > 1:
                        # SÓLO SI EL PUNTO LAS NOMBRA (3-oct-2026, AR 201 y
                        # 222/2025): en «confirma y niega» o «confirma, sobresee
                        # y concede» el punto no lleva a la responsable
                        # originaria y el aviso afirmaba algo falso. Se decide
                        # abajo, con los puntos ya elegidos.
                        _aviso_todas = (
                            "EL PUNTO DEL AMPARO NOMBRA A TODAS LAS AUTORIDADES DE LA "
                            "DEMANDA (" + "; ".join(_auts) + "): si el juzgado sobreseyó "
                            "respecto de alguna, quítala del punto resolutivo.")
            if not _orig:
                _orig = HUECO
                _avisos_bk.append(
                    "NO SE PUDO LEER LA AUTORIDAD RESPONSABLE ORIGINARIA y el "
                    "SEGUNDO punto resolutivo va con comodín. NO se puso el "
                    "órgano recurrido en su lugar: el amparo se concede o se "
                    "niega contra el acto de la autoridad que lo emitió en el "
                    "amparo indirecto, no contra el Juzgado de Distrito que lo "
                    "resolvió. Escríbela tú, está en la sentencia recurrida.")
            if _rama.get("aviso"):
                _avisos_bk.append(_rama["aviso"])
            # CONFIRMAR UNA CONCESIÓN ES REPRODUCIR SU RESOLUTIVO. La
            # fórmula genérica —«ampara y protege a X en términos del último
            # considerando de la resolución recurrida»— no dice contra qué
            # acto ni para qué efectos. Se reproduce el del juzgado, con la
            # cola apuntando a la sentencia recurrida, que es lo que pidió
            # David y lo que hace el precedente ARA 361/2025.
            _puntos = _rama["puntos"]
            if _clave == "confirma_concede":
                _parcial = _fr.hay_aspectos_no_combatidos(str(estudio or ""))
                _puntos = _ta.puntos_confirma_concede(
                    str(datos.get("resolutivo_recurrida") or ""),
                    parcial=_parcial)
                if _parcial:
                    _avisos_bk.append(
                        "EL PRIMER RESOLUTIVO DICE «EN LA MATERIA DE LA "
                        "REVISIÓN» porque el estudio afirma que algo de la "
                        "sentencia recurrida no fue combatido. Compruébalo: si "
                        "la recurrente sí lo impugnó todo, quita esa frase.")
            elif _tipo_reas == "concesion" and _clave in ("revoca_fondo_niega",
                                                          "revoca_fondo_concede"):
                # REVOCAR NO ES NEGAR (art. 93, fr. VI; AR 631/2025): el punto
                # del amparo sale del estudio de los conceptos no estudiados
                # —niega, concede por razón distinta, o hueco si no concluye—;
                # lo no impugnado queda firme y va PRIMERO, y la revocación se
                # acota a la materia de la revisión. El sujeto y el acto, los
                # del resolutivo del juzgado. Ver `tipos_asunto.puntos_reasuncion`.
                _sobresee_ad = (_que_hizo == "sobresee_concede"
                                or bool(_reas_d.get("sobresee_ademas")))
                # LO FIRME NO DEPENDE DE LA PROSA (AR 631/2025, al generar en
                # pantalla, 28-sep-2026): recurrió la tercera, la recurrida
                # también sobreseyó (en sus considerandos) y el estudio no lo
                # dijo; `_firme` salía falso y los resolutivos se quedaban en
                # «PRIMERO. En la materia de la revisión, se revoca…» sin el
                # punto del sobreseimiento. Si quien recurre no es la quejosa y
                # no consta adhesiva, nadie lo impugnó: queda firme por código
                # (`tipos_asunto.sobreseimiento_firme`, la regla de la ficha).
                _firme_prosa = _fr.declara_firme_el_sobreseimiento(_txt_est)
                _firme_codigo = (bool(_reas_d.get("sobreseimiento_firme"))
                                 or _ta.sobreseimiento_firme(_sobresee_ad, _quien_d,
                                                             bool(_reas_d.get("adhesiva"))))
                _firme = _firme_prosa or _firme_codigo
                _parcial = (_firme or _sobresee_ad
                            or _fr.hay_aspectos_no_combatidos(_txt_est))
                _puntos = _ta.puntos_reasuncion(
                    _sent_amparo, str(datos.get("resolutivo_recurrida") or ""),
                    firme=_firme, parcial=_parcial)
                _fund_r = _ta.FUNDAMENTO_REASUNCION["concesion"]
                if _sent_amparo in ("niega", "concede"):
                    _avisos_bk.append(
                        f"SE REVOCA LA CONCESIÓN Y SE REASUME JURISDICCIÓN ({_fund_r}): "
                        f"el estudio de los conceptos de violación que el juzgado no "
                        f"estudió concluye que procede "
                        f"{'NEGAR' if _sent_amparo == 'niega' else 'CONCEDER por una razón distinta'} "
                        f"el amparo, y el punto resolutivo lo dice. Compruébalo: "
                        + ("la negativa descansa en ese estudio, no en la sola revocación."
                           if _sent_amparo == "niega" else
                           "los efectos de esta concesión son los que fija el proyecto."))
                elif _reas_d.get("hacen_falta") == "por_confirmar" and not _reas_d.get("tenemos"):
                    _avisos_bk.append(
                        f"SE REVOCA UNA CONCESIÓN ({_fund_r}) Y LA SENTENCIA RECURRIDA NO DICE "
                        f"QUE EL JUZGADO DEJARA CONCEPTOS DE VIOLACIÓN SIN ESTUDIAR: si los "
                        f"examinó todos y la quejosa no combatió los desestimados, no hay nada "
                        f"que reasumir; si quedaron sin estudiar, hay que estudiarlos. El estudio "
                        f"no concluye si se concede o se niega y el punto resolutivo del amparo "
                        f"va con HUECO: compruébalo en la recurrida y complétalo.")
                elif _reas_d.get("tenemos") is False:
                    _avisos_bk.append(
                        f"SE REVOCA UNA CONCESIÓN Y LOS CONCEPTOS DE VIOLACIÓN QUE EL "
                        f"JUZGADO NO ESTUDIÓ NO CONSTAN: el tribunal tiene que estudiarlos "
                        f"({_fund_r}) y de ese estudio sale si se concede o se niega. El "
                        f"punto resolutivo del amparo va con HUECO: aporta los conceptos "
                        f"(o la demanda de amparo) y vuelve a generar, o complétalo tú "
                        f"después de estudiarlos. No firmes un «no ampara» sin ese estudio.")
                else:
                    _avisos_bk.append(
                        f"SE REVOCA LA CONCESIÓN Y SE REASUME JURISDICCIÓN ({_fund_r}), pero "
                        f"el estudio no dice si procede conceder o negar el amparo: el punto "
                        f"resolutivo del amparo va con HUECO. Complétalo con la conclusión "
                        f"del estudio de los conceptos no estudiados.")
                if _reas_d.get("donde") in ("constancias", "recurrida"):
                    _avisos_bk.append(
                        "LOS CONCEPTOS DE VIOLACIÓN NO ESTUDIADOS "
                        + _fr.DONDE_CONCEPTOS[_reas_d["donde"]].upper()
                        + ": comprueba que estén completos antes de firmar.")
                if _sobresee_ad and not _firme:
                    _avisos_bk.append(
                        "LA SENTENCIA RECURRIDA TAMBIÉN SOBRESEYÓ respecto de algún acto y el "
                        "proyecto no dice que ese sobreseimiento quedó firme. Si nadie lo "
                        "impugnó, dilo en el estudio y añade el punto «Queda firme el "
                        "sobreseimiento…»; la revocación ya va acotada a la materia de la "
                        "revisión."
                        + (" Consta revisión adhesiva: comprueba si la quejosa lo combate."
                           if _reas_d.get("adhesiva") else ""))
                elif _firme_codigo and not _firme_prosa:
                    # El punto va porque nadie pudo impugnarlo; que el estudio
                    # lo diga es cosa del secretario, y se le dice.
                    _avisos_bk.append(
                        "QUEDA FIRME EL SOBRESEIMIENTO que la sentencia recurrida decretó "
                        "respecto de otro acto: la recurrente no es la quejosa —la única a quien "
                        "ese sobreseimiento perjudica— y no consta revisión adhesiva, así que va "
                        "como PRIMER punto resolutivo. El estudio no lo dice: añade que ese "
                        "sobreseimiento no es materia de la revisión y queda firme.")
            elif (_clave == "revoca_fondo_concede" and _que_hizo == "concede"
                  and _quien_d.strip().lower() == "quejoso"):
                # LA QUEJOSA RECURRE SU CONCESIÓN Y GANA (revisión del 28-sep-2026,
                # AR 631/2025): la sentencia que corresponde la ampara (art. 93,
                # fr. V), con el sujeto y el acto del resolutivo del juzgado y
                # los efectos de esta ejecutoria. Revocar o sólo modificar lo
                # dice el estudio; si no lo dice, se revoca y se avisa.
                _modifica = bool(re.search(r"\bprocede\s+modificar\b|\bse\s+modifica\s+la\s+sentencia\b",
                                           _txt_est, re.I))
                _puntos = _ta.puntos_quejosa_mejora(
                    str(datos.get("resolutivo_recurrida") or ""),
                    "modifica" if _modifica else "revoca")
                _avisos_bk.append(
                    "RECURRE LA QUEJOSA CONTRA UNA CONCESIÓN Y SU AGRAVIO PROSPERA: el "
                    "resolutivo la ampara —su recurso no puede empeorarle la situación (art. "
                    "93, fr. V)— y "
                    + ("MODIFICA la sentencia recurrida, como dice el estudio."
                       if _modifica else
                       "REVOCA la sentencia recurrida. Si el vicio deja en pie el resto de "
                       "la concesión, lo que procede es MODIFICARLA: compruébalo.")
                    + " Los efectos son los que fija el proyecto.")
            elif _clave.startswith("revoca_sobreseimiento") and _que_hizo == "sobresee_concede":
                # LA QUEJOSA RECURRE UNA SENTENCIA MIXTA (revisión del 28-sep-2026):
                # combate el sobreseimiento; la concesión por los demás actos es
                # suya y nadie la recurrió. La revocación se acota.
                _puntos = [_ta.REVOCA_PARCIAL] + list(_puntos[1:])
                _avisos_bk.append(
                    "LA QUEJOSA RECURRE UNA SENTENCIA QUE SOBRESEYÓ UN ACTO Y LA AMPARÓ POR LOS "
                    "DEMÁS: se levanta el sobreseimiento y la revocación se acota a la materia de "
                    "la revisión; la concesión queda firme. Comprueba que el proyecto lo diga.")
            elif _clave == "revoca_fondo_niega":
                # REVOCAR UNA CONCESIÓN ES NEGAR LO QUE ELLA CONCEDIÓ, a quien
                # se lo concedió y contra el acto por el que lo concedió (AR
                # 631/2025: la fórmula genérica amparaba a la recurrente contra
                # un juzgado sobreseído). Ver `tipos_asunto.puntos_revoca_concesion`.
                _neg = _ta.puntos_revoca_concesion(
                    str(datos.get("resolutivo_recurrida") or ""))
                if _neg:
                    _puntos = _neg
                    _avisos_bk.append(
                        "EL SEGUNDO RESOLUTIVO NIEGA LO QUE EL JUZGADO CONCEDIÓ, con "
                        "el sujeto y el acto de su propio resolutivo. Compruébalo "
                        "contra la sentencia recurrida.")
            # ═══ LO QUE LA FICHA PROCESAL YA SABE (3-oct-2026, bandera) ═══
            # QUEDA FIRME EL SOBRESEIMIENTO QUE NADIE RECURRIÓ también cuando se
            # CONFIRMA: la recurrida sobreseyó un acto y amparó o negó por los
            # demás, y recurrió la autoridad o la tercera (a quienes ese
            # sobreseimiento no perjudica). Faltaba en 8 de 9 AR de octubre; el
            # circuito lo escribe así en 72 resolutivos de 2025-26: «Queda
            # firme…», «En la materia de la revisión, se confirma…», el amparo.
            if _rige_pt and _clave.startswith("confirma"):
                _sob_ad_c = (_que_hizo in ("sobresee_concede", "sobresee_niega")
                             or bool(_reas_d.get("sobresee_ademas")))
                if (bool(_reas_d.get("sobreseimiento_firme"))
                        or _ta.sobreseimiento_firme(_sob_ad_c, _quien_d,
                                                    bool(_reas_d.get("adhesiva")))):
                    _puntos = _puntos_con_firmeza(_puntos)
                    _avisos_bk.append(
                        "QUEDA FIRME EL SOBRESEIMIENTO que la sentencia recurrida "
                        "decretó respecto de otro acto: lo recurrió quien no es la "
                        "quejosa y no consta revisión adhesiva, así que va como PRIMER "
                        "punto y la confirmación se acota a la materia de la revisión. "
                        "Comprueba que el estudio lo diga.")
            # LOS EFECTOS Y LAS RAZONES REMITEN A ESTA EJECUTORIA POR SU ORDINAL
            # (3-oct-2026, bandera): «el último considerando de esta ejecutoria»
            # era ambiguo —los efectos van en el último considerando, que es el
            # del estudio, y la recurrida también tiene uno «último»—.
            _ord_ult = (_ORDINALES[min(len(con_apartados) - 1, 9)].lower()
                        if con_apartados else "")
            if _aviso_todas and any("{responsable_originaria}" in str(_pt) for _pt in _puntos):
                _avisos_bk.append(_aviso_todas)
            # ═══ EL PUNTO QUE CONFIRMA NOMBRA ACTO Y AUTORIDAD (C5, 3-oct-2026) ═══
            # David: «sí está bien que se precise… información valiosa para el
            # lector». Las ramas confirma_niega, confirma_concede y
            # confirma_sobresee traen {actos_del_amparo}: con la ficha, los actos
            # y las autoridades de la demanda de amparo indirecto, anclados al
            # papel (`procesal.actos`, `procesal.autoridades`), y la remisión al
            # resultando que los copia (`procesal.resultando_demanda`); sin ellos
            # —o sin la ficha, o sin la bandera—, la remisión de siempre
            # («respecto de / contra los actos precisados en la resolución
            # recurrida»). «contra» en el sobreseimiento, «respecto de» en las
            # demás (`tipos_asunto.con_actos_del_amparo`). Se llena ANTES que
            # {quejoso}, y lo que sigue (la suspensión, «en contra del», el
            # ordinal de «esta ejecutoria») corre sobre el texto ya lleno.
            _pro_a = _pro if (_rige_pt and isinstance(_pro, dict)) else {}
            _actos_pt = [a_ for a_ in (_pro_a.get("actos") or [])
                         if isinstance(a_, str) and _valor(a_)]
            _auts_pt = [_valor(a_) for a_ in (_pro_a.get("autoridades") or [])
                        if isinstance(a_, str) and _valor(a_)]
            _ord_dem = (_valor(_pro_a.get("resultando_demanda")) or "primero").lower()
            _pl_q = bool(_pro_a.get("plural_quejoso"))
            # UNA LISTA NUEVA, NUNCA LA DEL CATÁLOGO: `_puntos` puede ser la misma
            # lista de `RAMAS_REVISION`, y llenarla en su sitio dejaría los actos
            # de este asunto en el proyecto siguiente del mismo worker.
            _con_actos = False
            _puntos_llenos = []
            for _pt in _puntos:
                if "{actos_del_amparo}" in str(_pt):
                    _prep_pt = _ta.preposicion_del_amparo(_clave, str(_pt))
                    _lleno = _ta.punto_del_amparo(_actos_pt, _auts_pt, _ord_dem, _pl_q, _prep_pt)
                    _con_actos = _con_actos or bool(_lleno)
                    # SIN ACTOS, LA COLA DE SIEMPRE (revisión AR, 3-oct-2026): «…en la
                    # resolución recurrida, por las razones expuestas en el último
                    # considerando de la misma», no «resolución recurrida» dos veces.
                    _pt = (str(_pt).replace("{actos_del_amparo}", _lleno) if _lleno
                           else _ta.con_generico_del_amparo(str(_pt), _prep_pt))
                    # LA NORMA QUE LA FICHA NO TRAE (revisión AR, 3-oct-2026, AR
                    # 72/2025): con el Legislativo entre las responsables y ningún
                    # acto que sea norma, el punto remite y el secretario lo sabe.
                    if _ta.norma_sin_acto(_actos_pt, _auts_pt):
                        _av_norma = ("LA DEMANDA RECLAMA UNA NORMA QUE LA FICHA NO TRAE ENTRE LOS "
                                     "ACTOS (demanda.actos): el Poder Legislativo es autoridad "
                                     "responsable y ningún acto de la ficha es una norma general, así "
                                     "que el punto del amparo no nombra el acto ni la autoridad y "
                                     "remite a la resolución recurrida. Si el amparo se concedió o se "
                                     "negó contra la norma y su acto de aplicación, nómbralos en el "
                                     "punto.")
                        if _av_norma not in _avisos_bk:
                            _avisos_bk.append(_av_norma)
                _puntos_llenos.append(_pt)
            _puntos = _puntos_llenos
            if _con_actos and any(re.match(r"\s*PRIMERO\.\s+Queda\s+firme\b", str(_pt))
                                  for _pt in _puntos):
                _avisos_bk.append(
                    "EL PUNTO DEL AMPARO NOMBRA LOS ACTOS Y LAS AUTORIDADES DE LA DEMANDA, y la "
                    "sentencia recurrida sobreseyó respecto de alguno (queda firme): quita del punto "
                    "lo sobreseído, que ya no es materia de la revisión.")
            for _pt in _puntos:
                _txt = (_pt.replace("{HUECO}", HUECO)
                           .replace("{quejoso}", _q_prosa or HUECO)
                           .replace("{responsable_originaria}", _orig)
                           .replace("{expediente}",
                                    str(_datos_bk.get("expediente") or HUECO)))
                _txt = _contraer(_txt)
                if _rige_pt and _clase_ar in ("interlocutoria_suspension", "auto_sobreseimiento"):
                    _txt = _con_su_clase(_txt)
                    if _clase_ar == "interlocutoria_suspension":
                        # EN EL INCIDENTE NO SE AMPARA: se concede o se niega
                        # la suspensión definitiva (oro_AR: «SEGUNDO. Se
                        # [concede|niega] la suspensión definitiva solicitada
                        # por [quejoso]»).
                        _txt = re.sub(r"La Justicia de la Unión no ampara ni protege a ",
                                      "Se niega la suspensión definitiva solicitada por ", _txt)
                        _txt = re.sub(r"La Justicia de la Unión ampara y protege a ",
                                      "Se concede la suspensión definitiva solicitada por ", _txt)
                if _rige_pt:
                    # «EN CONTRA EL ACTO» (8 de 9 AR de octubre): el resolutivo
                    # del juzgado reproducido trae la preposición sin contraer.
                    _txt = re.sub(r"\ben\s+contra\s+el\b", "en contra del", _txt)
                    # SÓLO «EJECUTORIA»: es la palabra de las fórmulas de este
                    # tribunal; el punto del juzgado reproducido habla de «esta
                    # resolución» o «la sentencia recurrida», y sus efectos son
                    # los de SU último considerando, no los de éste.
                    if _ord_ult:
                        _txt = re.sub(
                            r"(?:el\s+)?(?:último\s+considerando|considerando\s+último)\s+de\s+"
                            r"(?:esta|la\s+presente)\s+ejecutoria",
                            f"el considerando {_ord_ult} de esta ejecutoria", _txt)
                _cab, _resto = _txt.split(". ", 1) if ". " in _txt else (_txt, "")
                _cab = _cab_de(_cab)
                # LOS PUNTOS RESOLUTIVOS LLEVAN SANGRÍA Y NO SE JUSTIFICAN.
                # Medido en el adelanto ajustado: sangría de primera línea y
                # alineación libre. Justificado, un resolutivo de una línea y
                # media queda con los huecos abiertos entre palabras que
                # delatan un documento mal compuesto, y es justo el párrafo que
                # todo el mundo mira.
                # JUSTIFICADO, Y EN LOS CUATRO TIPOS. Iba con `alineacion=None`
                # —«no la declares, que la herede del estilo»— porque así
                # estaba en el adelanto que David ajustó a mano. Medido en el
                # documento generado: los resolutivos salían SIN alineación, o
                # sea a la izquierda. David: «justifica el texto del
                # resolutivo, esto en todos los tipos de asuntos».
                _pr = tramos(doc, [(_cab + ". ", {"bold": True}), (_resto, {})],
                             sangria=True)
                # Media pulgada, que es lo que mide el suyo. La sangría del
                # cuerpo es 1.25 cm y la del resolutivo 1.27: no es un
                # descuido suyo, es el tabulador por omisión de Word.
                _pr.paragraph_format.first_line_indent = Cm(1.27)
            _avisos_bk.append(
                f"RESOLUTIVO DE REVISIÓN, rama «{_clave}» "
                f"({_rama['fundamento']}). El a quo "
                f"{_que_hizo or 'no consta qué resolvió'}; el recurso resultó "
                f"{'fundado' if concede else 'infundado'}. Compruébalo: de esto "
                f"depende que se confirme, se revoque o se modifique.")
            # EL SEGUNDO PUNTO NO SALE DEL RECURSO. Cuando se levanta el
            # sobreseimiento y el tribunal asume jurisdicción, el sentido del
            # amparo lo decide el estudio, y si el estudio no lo dijo, quien
            # firma tiene que saber que ahí se puso el sentido por omisión.
            if _rama.get("plenitud") and _que_hizo == "sobresee":
                _avisos_bk.append(
                    "SE ASUME JURISDICCIÓN Y EL SEGUNDO RESOLUTIVO "
                    + (f"NIEGA el amparo porque el estudio concluye que procede "
                       f"negarlo." if _sent_amparo == "niega" else
                       f"CONCEDE el amparo, que es el sentido por omisión "
                       f"porque el estudio no dice cuál es." if not _sent_amparo
                       else "CONCEDE el amparo, como concluye el estudio.")
                    + " Que el agravio sea fundado sólo prueba que no debió "
                      "sobreseerse: el fondo se estudia por primera vez aquí y "
                      "el sentido es tuyo.")
            _hecho = True
        except Exception as _erm:
            print(f"   ⚠️ rama de revisión no determinada: {_erm}")

    if _hecho:
        pass
    elif _res.get("punto"):
        # Queja y revisión: NO amparan. Se califica el recurso o se resuelve
        # sobre la sentencia recurrida, que es lo que hacen los engroses.
        _cal = _res["calif"][0] if concede else _res["calif"][1]
        # LOS DOS DATOS QUE IDENTIFICAN LA SENTENCIA. Se leen de los
        # resultandos, que es donde el propio proyecto acaba de escribirlos, y
        # si no se pudieron leer salen en hueco con su aviso: un resolutivo que
        # revoca «la sentencia recurrida» a secas no dice cuál, y quien lo
        # ejecute tiene que ir a buscarla.
        _fecha_s = _expte_s = ""
        _av_fecha = ""
        if "{fecha_sentencia}" in _res["punto"] and _pro:
            # CON LA FICHA, LOS DOS DATOS SON LOS SUYOS (3-oct-2026, bandera):
            # la fecha y el expediente de la sentencia de la Sala, nunca
            # `fecha_de`/`numero_de` sobre la prosa (RF_yucatan: «Se confirma la
            # sentencia de veintidós de abril», el día del auto de Presidencia).
            _fecha_s = _valor(_pro.get("fecha_acto"))
            _expte_s = _valor(_pro.get("expediente_tfja")) or _valor(_pro.get("expediente"))
            if not (_fecha_s and _expte_s):
                _avisos_bk.append(
                    "EL RESOLUTIVO NO IDENTIFICA LA SENTENCIA POR COMPLETO: "
                    + ("falta su FECHA. " if not _fecha_s else "")
                    + ("falta el EXPEDIENTE del juicio de nulidad. " if not _expte_s else "")
                    + "La ficha de trámite no lo trae y sale en hueco: escríbelo en "
                      "«Trámite en este tribunal» (fecha y expediente de la sentencia "
                      "recurrida) y vuelve a generar.")
        elif "{fecha_sentencia}" in _res["punto"]:
            try:
                import fase_origen as _fo_r
                _res_txt = " ".join(str(r.get("texto") or "")
                                    for r in (estructura.resultandos or []))
                _fuente = f"{datos.get('antecedentes') or ''} {_res_txt}"
                # PRIMERO LO LEÍDO DEL PDF. `fecha_de` y `numero_de`
                # leen la prosa del proyecto y están calibrados para eso;
                # sobre la sentencia de la Sala no aciertan, y su resultado
                # vacío dejaba el resolutivo con dos huecos. Medido sobre la
                # revisión fiscal 91/2025: del PDF salen el expediente
                # 695/25-09-01-7-OT y la fecha «veintidós de septiembre de dos
                # mil veinticinco»; de la prosa, nada.
                # LOS DOS LECTORES TIENEN QUE DECIR LO MISMO. Antes se
                # prefería el del PDF y, si discrepaba del cuerpo del
                # proyecto, ganaba en silencio: el 91/2025 salió con «Se
                # confirma la sentencia de TRES DE NOVIEMBRE» mientras sus
                # resultandos y considerandos la fechaban el VEINTIDÓS DE
                # SEPTIEMBRE cuatro veces. Ahora la discrepancia va a hueco con
                # las dos fechas escritas en el aviso.
                _fecha_s, _av_fecha = _fo_r.fecha_del_recurrido(
                    str(datos.get("fecha_origen") or ""), _fuente)
                if _av_fecha:
                    _avisos_bk.append(_av_fecha)
                _expte_s = (str(datos.get("expediente_origen") or "").strip()
                            or _fo_r.numero_de(_fuente))
            except Exception as _eo:
                print(f"   ⚠️ no se pudo leer fecha/expediente del recurrido: {_eo}")
            if (not _fecha_s and not _av_fecha) or not _expte_s:
                _avisos_bk.append(
                    "EL RESOLUTIVO NO IDENTIFICA LA SENTENCIA POR COMPLETO: "
                    + ("falta su FECHA. " if not _fecha_s and not _av_fecha
                       else "")
                    + ("falta el EXPEDIENTE de origen. " if not _expte_s else "")
                    + "No se pudo leer de los resultandos y sale en hueco. "
                      "Escríbelo: es lo que distingue esta sentencia de las "
                      "demás que dictó la misma Sala.")
        _texto = _res["punto"].format(
            calificacion=_cal,
            fecha_sentencia=_fecha_s or HUECO,
            expediente_origen=_expte_s or HUECO,
            responsable=_con_articulo(datos.get("responsable", "")) or HUECO)
        _texto = _contraer(_texto)
        # ═══════════════════════════════════════════════════════════════════
        # EN LA REVISIÓN FISCAL SÍ HAY REENVÍO
        # ═══════════════════════════════════════════════════════════════════
        # David: «a diferencia de la revisión en amparo indirecto, sí hay
        # reenvío porque la jurisdicción para el análisis del fondo corresponde
        # a la Sala Regional».
        #
        # Cuando el colegiado revoca y la Sala había dejado conceptos de
        # anulación sin estudiar, no puede sustituirla: se los devuelve. Un
        # resolutivo de un solo punto revoca y deja el juicio sin resolver y
        # sin nadie a quien le toque resolverlo.
        #
        # LAS DOS CONDICIONES, y las dos hacen falta. Que se REVOQUE, y que el
        # estudio diga que quedó algo sin estudiar. Revocar por un vicio formal
        # que no trasciende al sentido NO da lugar al reenvío: ahí el colegiado
        # corrige él mismo (registro 185493). Por eso no basta con mirar el
        # sentido del recurso.
        # SE IMPORTA AQUÍ. `_fr` se importa dentro de la rama del amparo en
        # revisión, que es OTRA rama de este mismo `if`: usarlo desde aquí es
        # el `UnboundLocalError` que ya dejó mudo el generador una vez. Lo cazó
        # el guardián al componer los cuatro tipos, no la lectura del código.
        import fase_rama as _fr_f
        # DÓNDE SE DICE QUE LA SALA OMITIÓ ALGO. No en el estudio: en el
        # RESUMEN DE LA SENTENCIA IMPUGNADA, que es donde se narra lo que la
        # Sala hizo. Medido sobre la revisión fiscal 91/2025: la frase «estudió
        # el tercer concepto de impugnación… y omitió estudiar los restantes»
        # está bajo el subtítulo «Sentencia impugnada», y pasándole sólo el
        # estudio el reenvío no disparaba y el resolutivo salía con un punto.
        #
        # NO SE MIRAN LOS AGRAVIOS. Ahí la omisión es lo que ALEGA la
        # recurrente, no lo que el tribunal encuentra; un agravio que se
        # declara infundado no da lugar a ningún reenvío.
        _fuente_omision = " ".join([
            " ".join(str(x) for x in (resumen_acto or [])),
            " ".join(str(x) for x in (estudio or []))
            if isinstance(estudio, (list, tuple)) else str(estudio or "")])
        _reenvio = concede and _fr_f.hay_conceptos_sin_estudiar(_fuente_omision)
        if _reenvio:
            tramos(doc, [(_cab_de("PRIMERO") + ". ", {"bold": True}), (_texto, {})],
                   sangria=False)
            _sala = _con_articulo(datos.get("responsable", "")) or HUECO
            tramos(doc, [(_cab_de("SEGUNDO") + ". ", {"bold": True}),
                         # DAVID: «en revisión la sentencia no se deja
                         # insubsistente, se revoca». El punto anterior ya la
                         # revocó: no queda nada que dejar insubsistente, y
                         # ordenarlo describe una potestad —la del amparo, en
                         # que el tribunal NO revoca el acto sino que manda a
                         # la responsable retirarlo— que aquí no se ejerce.
                         (f"Se ordena a {_sala} dictar otra sentencia en la "
                          f"que, con libertad de jurisdicción y siguiendo los "
                          f"lineamientos de esta ejecutoria, se ocupe de los "
                          f"conceptos de anulación cuyo estudio omitió.", {})],
                   sangria=False)
            _avisos_bk.append(
                "EL RESOLUTIVO ORDENA EL REENVÍO a la Sala, porque se revoca su "
                "sentencia y el estudio dice que dejó conceptos de anulación "
                "sin examinar. Es la diferencia con el amparo en revisión: aquí "
                "el colegiado NO asume jurisdicción, porque el estudio de esos "
                "conceptos corresponde en primera instancia a la Sala "
                "(registros 188742, 193181 y 196875). COMPRUEBA DOS COSAS: que "
                "el segundo punto ENUMERE qué quedó sin estudiar, y que el "
                "vicio de verdad trascienda al sentido del fallo —si es un "
                "error de forma que no trasciende, el colegiado lo corrige y no "
                "se reenvía (registro 185493)—.")
        else:
            tramos(doc, [(_cab_de("ÚNICO") + ". ", {"bold": True}), (_texto, {})],
                   sangria=False)
    else:
        formula = _AMPARA if concede else _NO_AMPARA
        # SE AMPARA A LA PARTE, y si compareció por representante se dice
        # «por conducto de su representante legal…»: la persona moral es la
        # que resiente el perjuicio; la física sólo la representa.
        import promovente as _pvr
        _rep_res = (_en_prosa(datos.get("representante")) if _nuevo
                    else str(datos.get("representante") or ""))
        # LA REPRESENTACIÓN QUE YA VA DENTRO DEL NOMBRE NO SE REPITE (3-oct-2026,
        # cuarta ronda; AD 335/2025: «la Sucesión a Bienes de X, a través de su
        # albacea Y» y además «, por conducto de su albacea Y»).
        if _nuevo and _rep_res and _plano_nombre(_rep_res) and \
                f" {_plano_nombre(_rep_res)} " in f" {_plano_nombre(_q_prosa)} ":
            _rep_res = ""
        # LA FIGURA, LA MISMA QUE EN LA LEGITIMACIÓN (3-oct-2026, quinta ronda,
        # F6): la del encargo y, si no la trae, la de la ficha; su forma la da
        # `tipos_asunto.figura_en_prosa` dentro de `por_conducto` (el «artículo
        # 12» con su ley y la coma antes del nombre, como el resultando). Sin
        # la ficha, la del encargo, como antes.
        _fig_res = str(datos.get("figura_representante") or "")
        if _nuevo and not _fig_res.strip():
            _fig_res = _sin_truncar((_ficha_t or {}).get("figura_representante"))
        _a_quien = _pvr.por_conducto(_q_prosa, _rep_res, _fig_res)
        # «a el Instituto…» → «al Instituto…»: con la ficha el nombre ya puede
        # traer su artículo (el Ejido, el Instituto como persona moral oficial).
        _a_prep = (f"al {_a_quien[3:]}" if (_nuevo and _a_quien.startswith("el "))
                   else f"a {_a_quien or HUECO}")
        # LOS EFECTOS SE NOMBRAN EN EL RESOLUTIVO (art. 74, fr. VI, LA: los
        # resolutivos expresan «cuando sea el caso, los efectos de la concesión
        # en congruencia con la parte considerativa»). Su propio banco lo mide:
        # «…y para los efectos precisados en el último considerando». Se
        # remite al considerando de Efectos por su ordinal.
        _ef_cola = ""
        if concede:
            _k_ef = next((i for i, (r_, _) in enumerate(con_apartados)
                          if str(r_).strip().rstrip(".").lower() == "efectos"), None)
            if _k_ef is not None:
                _ef_cola = (f", para los efectos precisados en el considerando "
                            f"{_ORDINALES[min(_k_ef, 9)].lower()} de la misma")
        _cola_ad = None
        if _rige_pt and _pro and _tn == "amparo_directo":
            # EL RESOLUTIVO DICE CUÁL SENTENCIA (3-oct-2026, bandera). «en contra
            # de la sentencia reclamada…, precisada en el primer resultando» no
            # la identifica: el resolutivo es lo que se ejecuta y se transcribe
            # en el oficio a la responsable. Con la ficha se escribe con los
            # mismos datos del V I S T O —fecha, órgano y toca o expediente—, en
            # la fórmula del banco (ad-resolutivos: «…contra la sentencia
            # dictada el {fecha}, por {responsable}, en el toca…»).
            _cola_ad = _cola_resolutivo_ad(_pro, _ficha_t, _avisos_bk)
            if concede and _cola_ad is not None:
                _k_ef2 = next((i for i, (r_, _) in enumerate(con_apartados)
                               if str(r_).strip().rstrip(".").lower() == "efectos"), None)
                if _k_ef2 is not None:
                    _cola_ad += (f", para los efectos precisados en el considerando "
                                 f"{_ORDINALES[min(_k_ef2, 9)].lower()} de esta ejecutoria")
        if _cola_ad is not None:
            tramos(doc, [(_cab_de("ÚNICO") + ". ", {"bold": True}),
                         ("La Justicia de la Unión ", {}),
                         (formula, {"bold": True}),
                         (f" {_a_prep}, {_cola_ad}.", {})],
                   sangria=False)
        else:
            tramos(doc, [(_cab_de("ÚNICO") + ". ", {"bold": True}),
                         ("La Justicia de la Unión ", {}),
                         (formula, {"bold": True}),
                         (f" {_a_prep}, en contra de "
                          f"{esq.get('recurrido','la sentencia reclamada')}, dictada "
                          f"por {_con_articulo(datos.get('responsable','')) or HUECO}, "
                          f"precisada en el primer resultando de esta ejecutoria"
                          f"{_ef_cola}.", {})],
                   sangria=False)

    # EL PUNTO DEL ADHESIVO, DETRÁS DE LOS DEMÁS (3-oct-2026, bandera; oro_AD:
    # «PRIMERO. … ampara … SEGUNDO. Se declara sin materia el amparo adhesivo»).
    if _adh_res:
        tramos(doc, [(f"{_ORDINALES[min(_n_puntos[0], 9)]}. ", {"bold": True}),
                     (_adh_res, {})], sangria=False)

    # LA QUEJA DE LA FRACCIÓN II NO VIENE DE UN JUZGADO (3-oct-2026, bandera):
    # el auto recurrido lo dictó la autoridad responsable en un amparo directo
    # radicado en este tribunal, y a ella va el testimonio. «Envíese testimonio
    # … al juzgado de origen» mandaba la resolución a un juzgado que no existe
    # en el asunto.
    _notif = _res["notif"]
    if _rige_pt and _tn == "queja" and _fr97 == "II":
        _notif = _notif.replace("al juzgado de origen", "a la autoridad responsable")
    elif _fr97_en_hueco:
        # SIN FRACCIÓN NO SE SABE SI ES UN JUZGADO (F3): el testimonio va a quien
        # dictó el auto, que es verdad en las dos.
        _notif = _notif.replace("al juzgado de origen", "al órgano que dictó el auto recurrido")
    parrafo(doc, _notif, sangria=True)

    # DAVID: «en lugar de poner los dos nombres hasta abajo del proyecto del
    # secretario y del magistrado, eso no sirve, hay que generar esta portada
    # del proyecto que trae la síntesis».
    #
    # No se pierde ningún dato: los dos nombres siguen en la carátula, arriba,
    # que es donde el lector los busca. Abajo sólo se repetían.
    # ═══════════════════════════════════════════════════════════════════════
    # LA HOJA QUE CIRCULA NO PUEDE PROPONER LO QUE EL RESOLUTIVO NIEGA
    # ═══════════════════════════════════════════════════════════════════════
    # La SÍNTESIS es la hoja que se desprende y encabeza la carpeta que se
    # circula a los magistrados. Con el estudio en reserva, `fase_sintesis`
    # sólo ve el estudio de fondo —nunca el cómputo— y escribía «Propuesta de
    # resolución: se propone determinar que…» encima de una ejecutoria cuyo
    # único resolutivo desecha por extemporáneo. Lo que circulaba del proyecto
    # era, literalmente, la propuesta contraria a su resolutivo: la
    # incongruencia del artículo 74, fracción VI, en la primera página.
    #
    # Su propia regla de escape —devolver el título vacío si el recurso se
    # desechó— no puede dispararse, porque el estudio de fondo nunca habla de
    # la oportunidad. Así que la guarda va aquí, donde sí se sabe.
    # SIN FIRMAS AL PIE. David, 16-sep-2026: «los nombres de magistrado y
    # secretario no tienen que ir hasta abajo, quita esa regla». Ya están en la
    # carátula, que es donde se leen, y el engrose los firma cuando se firma:
    # el proyecto que sale del taller es un borrador para trabajar, y un pie de
    # firmas invita a tratarlo como si ya estuviera.
    if not (_extemp or _cumpl_sob):
        _bloque_sintesis(doc, sintesis or {})

    # ═══════════════════════════════════════════════════════════════════════
    # EL ESTUDIO EN RESERVA — DETRÁS DE LOS RESOLUTIVOS, FUERA DE LA EJECUTORIA
    # ═══════════════════════════════════════════════════════════════════════
    # David: «nunca impedir el estudio de fondo si el secretario decide generar
    # proyecto de fondo». Cuando él ACEPTA la extemporaneidad y aun así lo
    # quiere, el estudio no puede ir dentro de la ejecutoria: el análisis de
    # la improcedencia es oficioso y preferente (artículo 62 de la Ley de
    # Amparo) y una ejecutoria que sobresee y además estudia el fondo es la
    # incongruencia que costó el proyecto del comentario de main.py.
    #
    # Y NO SE ESCRIBE EN HIPOTÉTICO. Medido sobre los 40 engroses reales del
    # corpus con texto de oro: CERO usan «en el supuesto de que se considerara
    # oportuna», «suponiendo sin conceder» ni ninguna de sus variantes. El
    # tribunal de este corpus no razona en condicional; resuelve. Así que el
    # estudio se escribe en los mismos términos categóricos de siempre y lo
    # que cambia es DÓNDE está y qué dice su cabecera.
    #
    # No se marca con color ni con cursiva: se marca con un salto de página, un
    # rótulo y una frase que dice qué es. Se borra seleccionando de aquí al
    # final.
    if _reserva:
        # EL SALTO VA EN EL RÓTULO, NO EN UN PÁRRAFO APARTE. `add_page_break()`
        # añade un párrafo vacío, y ese párrafo cae DENTRO de la ejecutoria: la
        # comprobación de que la ejecutoria con anexo es idéntica a la de hoy
        # fallaba por él. Con `page_break_before` el salto es del rótulo del
        # anexo y la ejecutoria no se entera.
        _rot_anexo = rotulo(doc, "Anexo de trabajo")
        try:
            _rot_anexo.paragraph_format.page_break_before = True
        except Exception:
            pass
        _ex_r = _ta.extemporaneo_de(tipo_asunto)
        # «CONFORME AL ARTÍCULOS 61, FRACCIÓN XIV» NO ES ESPAÑOL. El catálogo
        # trae unas veces «artículo» y otras «artículos», y la preposición
        # cambia con el número. El aviso de arriba arrastra ese defecto desde
        # que se escribió; aquí no puede arrastrarse, porque esto sale EN EL
        # PAPEL.
        import fase0_oportunidad as _f0c
        _conf = _f0c.conforme_a(_ex_r["fundamento"])
        # NO SE LE PONE EN LA BOCA LO QUE NO DIJO. La reserva puede venir de
        # dos sitios: pedida por quien proyecta —con su razón escrita— o
        # aplicada sola porque calificó el fondo pese al aviso. En el segundo
        # caso no hubo petición ni razón declarada, y el anexo no puede
        # afirmar que las hubo: el .docx lo firma una persona.
        _auto_r = bool(getattr(computo, "decision_automatica", False))
        parrafo(doc,
                f"ESTE APARTADO NO FORMA PARTE DE LA EJECUTORIA. La ejecutoria "
                f"que antecede propone resolver la improcedencia por "
                f"extemporaneidad, conforme {_conf}, y ése es "
                f"su único punto resolutivo. El estudio que sigue se agrega "
                + ("porque se calificó el fondo pese al cómputo de "
                   "extemporaneidad" if _auto_r else
                   "a petición de quien proyecta")
                + f", para que el Pleno cuente con él "
                f"si no comparte esa conclusión sobre la oportunidad. No se "
                f"somete a votación, no se notifica a las partes y debe "
                f"suprimirse antes de listar el asunto si la extemporaneidad "
                f"se confirma.")
        _mot_r = (getattr(computo, "motivo", "") or "").strip()
        if _mot_r and not _auto_r:
            parrafo(doc, f"Razón declarada por quien proyecta: «{_mot_r}».")
        elif _auto_r:
            parrafo(doc,
                    "Quien proyecta no ha declarado razón para apartarse del "
                    "cómputo. Si sostiene que la presentación fue oportuna, "
                    "debe decirlo y escribir por qué: entonces el estudio "
                    "entra en la ejecutoria y este anexo desaparece.")
        _subtitulo(doc, "Estudio de fondo, en reserva")
        _cuerpo_anexo = (list(_sin_remate_duplicado(_cuerpo_estudio))
                         + list(_sin_remate_duplicado(_cuerpo_conceptos)))
        if _cuerpo_anexo:
            _escribir_estudio(doc, _cuerpo_anexo, tesis, notas, normas)
        else:
            parrafo(doc, "No se escribió estudio de fondo para este asunto.")
            _avisos_bk.append(
                "PEDISTE EL ESTUDIO EN RESERVA Y NO HAY ESTUDIO QUE PONER: el "
                "anexo sale vacío. Comprueba que dictaste el criterio antes de "
                "resolver.")
        if _efectos_escritos:
            _subtitulo(doc, "Efectos, en reserva")
            for _x in _efectos_escritos:
                if str(_x).strip():
                    parrafo(doc, str(_x).strip())

    # RED DE SEGURIDAD. Si alguna marca sobrevivió a todo lo anterior —porque el
    # modelo la escribió de una forma que no previmos—, se borra antes de
    # guardar. El andamio no sale al papel, y punto.
    _RX_RESTO = re.compile(r"\s*\[{1,2}[^\[\]]{0,120}\]{0,2}")
    for p in doc.paragraphs:
        if "[[" in p.text:
            entero = _RX_RESTO.sub("", p.text)
            if p.runs:
                p.runs[0].text = entero
                for r in p.runs[1:]:
                    r.text = ""
    # ═══════════════════════════════════════════════════════════════════════
    # LA VERJA PROCESAL (3-oct-2026, bandera `procedencia_por_tipo`)
    # ═══════════════════════════════════════════════════════════════════════
    # Antes de entregar, una revisión determinista del bloque procesal —de la
    # carátula al rótulo de los Antecedentes o del Estudio, y los
    # resolutivos—: huecos fuera de la sesión, fórmulas evasivas, fechas que no
    # están en la ficha, la responsable escrita de dos formas, el supletorio
    # que no es de la sede… (`verja_procesal.revisar`). Primero su arreglo de
    # tipografía segura («de el» → «del», «5º» → «5o.»), corrida a corrida para
    # no tocar las negritas; después la revisión, sobre el texto ya arreglado.
    # SUS AVISOS VAN PRIMERO: son los que dicen si lo procesal se puede firmar,
    # y Gemini ya no lo revisa.
    _avisos_verja: list = []
    if _rige_pt:
        try:
            import verja_procesal as _vp
            _ps_doc = list(doc.paragraphs)
            _txts = [p_.text for p_ in _ps_doc]
            _rot_est = _ta.rotulo_estudio_de(tipo_asunto).strip().rstrip(".")
            _rx_fin = re.compile(
                r"^(?:" + "|".join(_ORDINALES) + r")\.\s+(?:Antecedentes\.|"
                + re.escape(_rot_est) + r"\.)")
            _fin_p = next((i for i, t_ in enumerate(_txts)
                           if _rx_fin.match(t_) or t_.startswith("Por lo expuesto y fundado")),
                          len(_txts))
            _ini_r = next((i for i, t_ in enumerate(_txts)
                           if t_.replace(" ", "") == "RESUELVE"), None)
            _fin_r = (next((i for i in range(_ini_r, len(_txts))
                            if _txts[i].startswith("Notifíquese")), len(_txts) - 1)
                      if _ini_r is not None else None)
            # LA ANTESALA DE LOS RESOLUTIVOS ENTRA (3-oct-2026): el bloque se
            # cortaba en «Por lo expuesto y fundado» y la verja no veía el
            # párrafo que lo trae, que es donde el corpus escribe la cadena de
            # la Ley Orgánica abrogada («37, fracción V», 13 revisiones
            # fiscales). Se empieza en el último «Por lo expuesto…» antes del
            # R E S U E L V E, si lo hay.
            if _ini_r is not None:
                _ante = next((i for i in range(_ini_r - 1, max(_fin_p, 0) - 1, -1)
                              if _txts[i].startswith("Por lo expuesto")), None)
                if _ante is None and _ini_r and _txts[_ini_r - 1].startswith("Por lo expuesto"):
                    _ante = _ini_r - 1
                if _ante is not None:
                    _ini_r = _ante
            _rango = sorted(set(range(0, _fin_p)) | (
                set(range(_ini_r, _fin_r + 1)) if _ini_r is not None else set()))
            for i in _rango:
                for r_ in _ps_doc[i].runs:
                    if r_.text:
                        _t2, _c2 = _vp.arreglar(r_.text)
                        if isinstance(_t2, str) and _t2 != r_.text:
                            r_.text = _t2
            _texto_proc = "\n".join(_ps_doc[i].text for i in _rango)
            # CON EL DETALLE, SI LA VERJA LO DA: de qué regla es cada aviso, para
            # no repetir el hueco que otra pieza ya avisó (ver
            # `_fundir_avisos_de_hueco`). Una verja sin detalle, como antes.
            _rv_det = getattr(_vp, "revisar_detalle", None)
            # LO QUE OTRAS PIEZAS YA AVISARON va a la verja (`avisos_previos`)
            # para que ella misma no repita el hueco cuya clave ya está dicha.
            _previos_v = _avisos_de_fuera + list(avisos_doc) + list(_avisos_bk)
            if callable(_rv_det):
                try:
                    _det_v = _rv_det(_texto_proc, _ficha_t, datos, _tn, avisos_previos=_previos_v)
                except TypeError:
                    _det_v = _rv_det(_texto_proc, _ficha_t, datos, _tn)
                _det_v = [d_ for d_ in (_det_v or []) if isinstance(d_, dict)]
            else:
                _det_v = [{"regla": "", "aviso": str(a_)}
                          for a_ in (_vp.revisar(_texto_proc, _ficha_t, datos, _tn) or [])]
            # (g) EL SUPLETORIO EN TODO EL DOCUMENTO (3-oct-2026, revisión de
            # fundamentos; SPEC §3.3 g: «en CUALQUIER parte del documento»). Los
            # Antecedentes y el Estudio citan el hecho notorio y la prueba
            # electrónica con su código supletorio, y la verja sólo veía el
            # bloque procesal: un «artículo 269 del Código Nacional…» en el
            # estudio de un proyecto de Querétaro no se acusaba. Lo que queda
            # entre el bloque procesal y la antesala de los resolutivos pasa
            # por esa regla sola, como un considerando más.
            _fin_resto = (_ini_r if _ini_r is not None else len(_txts))
            _resto = "\n".join(_txts[i] for i in range(_fin_p, _fin_resto) if _txts[i].strip())
            if _resto.strip():
                _texto_resto = "C O N S I D E R A N D O\n" + _resto
                _rg_sup = getattr(_vp, "_regla_supletorio", None)
                _secs_f = getattr(_vp, "secciones", None)
                _g_vistos = {str(d_.get("aviso")) for d_ in _det_v}
                _nuevos_g = []
                try:
                    if callable(_rv_det):
                        try:
                            _nuevos_g = _rv_det(_texto_resto, _ficha_t, datos, _tn, reglas="g")
                        except TypeError:
                            _nuevos_g = _rv_det(_texto_resto, _ficha_t, datos, _tn)
                        _nuevos_g = [d_ for d_ in (_nuevos_g or [])
                                     if isinstance(d_, dict) and d_.get("regla") == "g"]
                    elif callable(_rg_sup) and callable(_secs_f):
                        _out_g: list = []
                        _rg_sup(_texto_resto, _secs_f(_texto_resto), _ficha_t, datos, _out_g)
                        _nuevos_g = [{"regla": "g", "aviso": str(x_[1] if isinstance(x_, tuple) else x_)}
                                     for x_ in _out_g]
                except Exception:
                    _nuevos_g = []
                _det_v += [d_ for d_ in _nuevos_g if str(d_.get("aviso")) not in _g_vistos]
            _avisos_verja = _fundir_avisos_de_hueco(
                _det_v, _avisos_de_fuera + list(avisos_doc) + list(_avisos_bk))
            _avisos_verja = [a_ for a_ in _avisos_verja if str(a_ or "").strip()]
        except Exception as _ev:
            _avisos_verja = [
                f"LA VERJA PROCESAL NO PUDO CORRER ({type(_ev).__name__}): el bloque "
                f"de procedencia —resultandos, competencia, existencia, legitimación, "
                f"oportunidad y resolutivos— sale sin su revisión determinista. "
                f"Revísalo a mano antes de firmar."]
    _airear(doc)
    doc.save(ruta_salida)
    _inyectar_notas(ruta_salida, notas)
    # Los avisos deterministas de la carátula viajan con el documento. Se
    # cuelgan de la estructura porque es lo que ya recorre el camino de vuelta.
    try:
        for _a in list(_avisos_verja) + list(avisos_doc) + list(_avisos_bk) + list(avisos_cotejo):
            if _a not in estructura.avisos:
                estructura.avisos.append(_a)
                estructura.avisos_de_composicion.append(_a)
        if _avisos_verja:
            estructura.avisos = (list(_avisos_verja)
                                 + [a_ for a_ in estructura.avisos if a_ not in _avisos_verja])
        # UN HECHO, UN AVISO (cuarta ronda, E11): con la ficha, lo que validar,
        # el compositor, la verja y este archivo dijeron dos veces, una.
        if _nuevo:
            estructura.avisos = _un_aviso_por_hecho(estructura.avisos,
                                                    estructura.avisos_de_composicion)
            estructura.avisos_de_composicion = [a_ for a_ in estructura.avisos_de_composicion
                                                if a_ in estructura.avisos]
    except Exception:
        pass
    return ruta_salida
