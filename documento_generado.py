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
import os
import re
from dataclasses import dataclass, field

import docx
from docx.enum.table import WD_TABLE_ALIGNMENT
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
TAMANO_CITA = Pt(12)
TAMANO_TABLA = Pt(11)
SANGRIA = Cm(1.25)
# La sangría del bloque transcrito —artículo o tesis—, medida en el corpus.
SANGRIA_CITA = Cm(1.25)
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

def _fmt(p, sangria=True, tamano=TAMANO, interlineado=INTERLINEADO,
         alineacion=WD_ALIGN_PARAGRAPH.JUSTIFY):
    pf = p.paragraph_format
    pf.alignment = alineacion
    pf.line_spacing = interlineado
    pf.space_after = Pt(6)
    pf.first_line_indent = SANGRIA if sangria else Cm(0)
    for r in p.runs:
        r.font.name = FUENTE
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


def rotulo(doc, texto):
    """«R E S U L T A N D O:» — centrado, en negrita y espaciado."""
    p = doc.add_paragraph()
    r = p.add_run(" ".join(texto.upper()))
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
        piezas.append(
            f'<w:footnote w:id="{i}"><w:p><w:pPr>'
            f'<w:spacing w:after="0" w:line="240" w:lineRule="auto"/>'
            f'<w:jc w:val="both"/></w:pPr>'
            f'<w:r><w:rPr><w:rStyle w:val="FootnoteReference"/>'
            f'<w:vertAlign w:val="superscript"/></w:rPr>'
            f'<w:footnoteRef/></w:r>'
            f'<w:r><w:rPr><w:rFonts w:ascii="{FUENTE}" w:hAnsi="{FUENTE}"/>'
            f'<w:sz w:val="18"/></w:rPr>'
            f'<w:t xml:space="preserve"> {_esc(texto)}</w:t></w:r>'
            f'</w:p></w:footnote>')
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


def tesis_del_rubro(texto: str, tesis: list):
    """La tesis del acervo cuyo rubro se cita en este párrafo, si alguna."""
    m = _RX_RUBRO.search(texto or "")
    if not m:
        return None, None
    citado = _normaliza_rubro(m.group(1))
    if len(citado) < 25:
        return None, None
    for t in (tesis or []):
        real = _normaliza_rubro(t.get("rubro", ""))
        if real and (real.startswith(citado[:70]) or citado.startswith(real[:70])):
            return t, m
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
    r"(?:jurisprudencia|tesis|criterio)\b", re.I | re.S)

# Los verbos de enlace que sabemos leer. Si el modelo escribe otra cosa, se usa
# el suyo tal cual mientras no nombre instancia ni tipo; y si no hay nada
# aprovechable, «Sirve de apoyo», que es la fórmula del oficio.
_POR_DEFECTO = "Sirve de apoyo"


# Lo que puede quedar colgando al recortar el sintagma: el modelo escribe
# «Sirve de apoyo, como criterio orientador, la jurisprudencia…» y el recorte se
# lleva desde «criterio», dejando «Sirve de apoyo, como». Compuesto luego da
# «Sirve de apoyo, como, la jurisprudencia…», con la palabra huérfana entre
# comas. Se poda la cola.
_RX_COLA_HUERFANA = re.compile(
    r"[\s,;:]*\b(como|en\s+calidad\s+de|a\s+t[íi]tulo\s+de|con\s+car[áa]cter"
    r"|por|de|del|la|el|los|las|y|e)\s*$", re.I)


def _verbo_de_enlace(anuncio: str) -> str:
    """Lo único del anuncio que escribe el modelo y merece conservarse."""
    a = " ".join((anuncio or "").split())
    if not a:
        return _POR_DEFECTO
    m = _RX_VERBO.match(a)
    if m and m.group(1).strip():
        v = m.group(1).strip().rstrip(",;:")
        # Se poda dos veces: «Sirve de apoyo, como» → «Sirve de apoyo»; y un
        # «Resulta aplicable, en calidad de» → «Resulta aplicable».
        for _ in range(2):
            v = _RX_COLA_HUERFANA.sub("", v).strip()
        return v or _POR_DEFECTO
    # Sin sustantivo reconocible: se conserva sólo si es corto y no nombra
    # órgano ni tipo, que es lo que no puede venir de él.
    if len(a) <= 60 and not re.search(
            r"sala|pleno|colegiado|jurisprudencia|tesis aislada", a, re.I):
        return a.rstrip(" ,;:")
    return _POR_DEFECTO


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
    if t.get("obligatoria"):
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
    return frase + ", de rubro y texto siguientes:"


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
    p.paragraph_format.keep_with_next = True

    # El texto ÍNTEGRO, en cursiva y desde el acervo.
    cuerpo = (t.get("texto") or "").strip()
    if cuerpo:
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
    if inst or loc:
        z = doc.add_paragraph()
        rz = z.add_run(inst.upper() + ("." if inst else ""))
        rz.bold = True
        _fmt(z, sangria=False, tamano=TAMANO_CITA,
             interlineado=INTERLINEADO_CITA)
        z.paragraph_format.left_indent = Cm(1.25)
        z.paragraph_format.keep_with_next = False
        if loc or reg:
            pie = loc if loc else ""
            if reg and reg not in pie:
                pie = (pie + ", " if pie else "") + f"registro digital {reg}"
            if pie in notas:
                _run_llamada(z, notas.index(pie) + 1)   # se reusa la existente
            else:
                notas.append(pie)
                _run_llamada(z, len(notas))


# ═══════════════════════════════════════════════════════════════════════════
# LA TABLA DEL CÓMPUTO
# ═══════════════════════════════════════════════════════════════════════════

def tabla_computo(doc, computo, fecha_en_letra) -> None:
    """El cómputo del plazo, en negro y gris.

    No es adorno: es la parte de la sentencia que más se revisa y la que peor
    se lee en prosa. Una fila por hito, la fecha al lado, y el resultado
    destacado abajo. Quien la revisa comprueba en diez segundos lo que en un
    párrafo corrido cuesta releer tres veces.
    """
    filas = [
        ("Notificación de la sentencia reclamada",
         fecha_en_letra(computo.notificacion)),
        (f"Surtimiento de efectos ({computo.regla.descripcion})",
         fecha_en_letra(computo.surtio)),
        ("Inicio del plazo", fecha_en_letra(computo.inicio)),
        (f"Plazo legal", f"{computo.plazo} días hábiles"),
        ("Vencimiento del plazo", fecha_en_letra(computo.vencimiento)),
    ]
    if computo.presentacion is not None:
        filas.append(("Presentación de la demanda",
                      fecha_en_letra(computo.presentacion)))
    if computo.inhabiles_en_medio:
        # LA FILA MENTÍA. Decía «Días inhábiles descontados: 1» mientras el
        # párrafo explicaba que se descontaron seis sábados y domingos más un
        # festivo. Contaba SÓLO los inhábiles declarados, no los fines de
        # semana. Se dice lo que de verdad cuenta.
        filas.append(("Días inhábiles declarados (además de sábados y domingos)",
                      str(len(computo.inhabiles_en_medio))))

    t = doc.add_table(rows=1, cols=2)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    _bordes(t)

    _celda(t.rows[0].cells[0], "CÓMPUTO DEL PLAZO", negrita=True,
           color=BLANCO, fondo=GRIS_CABECERA)
    _celda(t.rows[0].cells[1], "FECHA", negrita=True, color=BLANCO,
           fondo=GRIS_CABECERA, alineacion=WD_ALIGN_PARAGRAPH.CENTER)

    for i, (concepto, valor) in enumerate(filas):
        fila = t.add_row()
        fondo = GRIS_ALTERNO if i % 2 == 0 else None
        _celda(fila.cells[0], concepto, fondo=fondo)
        _celda(fila.cells[1], valor, fondo=fondo,
               alineacion=WD_ALIGN_PARAGRAPH.CENTER)

    if computo.oportuna is not None:
        fila = t.add_row()
        veredicto = ("PRESENTADA EN TIEMPO" if computo.oportuna
                     else "PRESENTADA FUERA DE PLAZO")
        _celda(fila.cells[0], "Resultado", negrita=True, color=BLANCO,
               fondo=GRIS_CABECERA)
        _celda(fila.cells[1], veredicto, negrita=True, color=BLANCO,
               fondo=GRIS_CABECERA, alineacion=WD_ALIGN_PARAGRAPH.CENTER)

    for fila in t.rows:
        fila.cells[0].width = Cm(9.5)
        fila.cells[1].width = Cm(5.0)


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
    _identifica = _IDENTIFICA_ACTO.get(_tipo, _IDENTIFICA_ACTO["amparo_directo"])
    return f"""Eres el secretario de un Tribunal Colegiado de Circuito y escribes las
partes ESTRUCTURALES de una sentencia de {_clase}. No escribes el estudio
de fondo —ese ya está hecho—: escribes lo que la ley obliga a decir antes de
llegar a él, con los datos de ESTE asunto y de ESTE tribunal.

EL TRIBUNAL QUE RESUELVE: {datos.get('tribunal','')}
CIUDAD: {datos.get('ciudad','')}
EXPEDIENTE: {datos.get('encabezado','')}
QUEJOSO: {datos.get('quejoso','')}
AUTORIDAD RESPONSABLE: {datos.get('responsable','')}
ACTO RECLAMADO: {datos.get('acto','(consta en los antecedentes)')}
FECHA DE PRESENTACIÓN DE LA DEMANDA: {datos.get('presentacion','')}
MAGISTRADO PONENTE: {datos.get('magistrado','')}
SECRETARIO: {datos.get('secretario','')}

ANTECEDENTES DEL ASUNTO, ya redactados
{datos.get('antecedentes','')[:4000]}

REGLAS:
- ESTOS SON LOS ROTULOS MEDIDOS EN 26 ENGROSES DE ESTE TRIBUNAL, no una
  propuesta: escríbelos tal cual. El resultando lleva esos cuatro apartados y
  el documento añade solo el de la sesión.
- NO ESCRIBAS ORDINALES. Nada de «PRIMERO.» ni «SEGUNDO.»: el documento los
  calcula y ponerlos aquí los duplica.
- NO INVENTES DATOS. Si no sabes una fecha, un número de toca o un nombre, NO
  lo pongas y NO lo sustituyas por uno verosímil: redacta la frase de modo que
  no lo necesite, o deja constancia de que consta en autos. Un dato inventado
  en un resultando se firma.
- EL TRIBUNAL ES EL DE ARRIBA, no otro. La competencia se funda en los
  artículos 103, fracción I, y 107, fracción V, de la Constitución; 33,
  fracción II, 34 y 170, fracción I, de la Ley de Amparo; y 37 de la Ley
  Orgánica del Poder Judicial de la Federación, con el acuerdo general que
  fije la jurisdicción territorial de ese tribunal —si no sabes cuál es, no
  cites acuerdo alguno—.
- FRASE de unas 35 palabras, subordinada. Voz impersonal: «se estima», «este
  Tribunal Colegiado considera». Nunca primera persona del singular.
- Sin Markdown, sin viñetas.

Devuelve SÓLO este JSON:
{{"apertura": "<ciudad y fórmula de resolución del tribunal; la FECHA DE LA SESIÓN se deja como ___ porque aún no ocurre>",
  "visto": "<la fórmula VISTO, para resolver el juicio de amparo directo…, sin repetir el rótulo>",
  "resultandos": [
     {{"titulo": "Presentación de la demanda de amparo",
       "texto": "<fecha, oficialía, promovente y su carácter. Después, IDENTIFICA el acto: {_identifica}. Si alguno de esos datos NO consta, NO lo menciones ni expliques que no consta: se omite y ya. PROHIBIDO resumir aquí su razonamiento: eso va en el estudio>"}},
     {{"titulo": "Derechos humanos cuya violación se alega", "texto": "<UNA sola frase con la lista de artículos constitucionales. No argumenta>"}},
     {{"titulo": "Tercero interesado", "texto": "<una frase: le resulta tal carácter a X, quien fue emplazado al presente juicio, según las constancias>"}},
     {{"titulo": "Trámite del juicio de amparo", "texto": "<auto de Presidencia, registro, admisión, vista del artículo 181 de la Ley de Amparo, y que el agente del Ministerio Público adscrito omitió formular pedimento>"}}
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
    r = await cliente.chat.completions.create(**kw)
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
                         es_recurso: bool = False) -> str:
    """El marco jurídico, escrito. Devuelve texto vacío si no hay material."""
    if not (material_marco or "").strip():
        return ""
    q = "agravios" if es_recurso else "conceptos de violación"
    lista = "\n".join(
        f"- {p.get('pregunta','') if isinstance(p, dict) else str(p)}"
        for p in (problemas or []))
    prompt = f"""Eres el secretario de un Tribunal Colegiado y escribes el apartado
de MARCO JURÍDICO de una sentencia de amparo: la premisa mayor con la que
después se resolverán los planteamientos.

LOS PROBLEMAS QUE HAY QUE RESOLVER
{lista}

MATERIAL RECUPERADO DEL BLOQUE DE CONSTITUCIONALIDAD Y DE LA LEY LOCAL
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
- EL PRECEPTO LOCAL O SECUNDARIO decisivo SÍ se transcribe, entre comillas y
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
  posible dar solución a los planteamientos de la parte quejosa.»

Devuelve SÓLO el texto del apartado, en párrafos separados por una línea en
blanco. Sin rótulo ni encabezado: el documento se lo pone."""
    kw = dict(model=MODELO_ESTRUCTURA,
              max_completion_tokens=MAX_TOKENS_ESTRUCTURA,
              messages=[{"role": "user", "content": prompt}])
    if ESFUERZO_ESTRUCTURA:
        kw["reasoning_effort"] = ESFUERZO_ESTRUCTURA
    r = await cliente.chat.completions.create(**kw)
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
        "punto": "Se {calificacion} la sentencia recurrida, dictada por {responsable}.",
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
_CONECTIVAS = {"de", "del", "la", "las", "el", "los", "y", "en", "e", "al"}


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
    for i, w in enumerate(n.split()):
        limpio = w.strip(".,;:()")
        if _romano.match(limpio.upper()) and limpio.upper() == limpio:
            partes.append(w)                       # ya viene en versales
        elif _romano.match(limpio.upper()) and len(limpio) > 1:
            partes.append(w.upper())               # «ii» → «II»
        elif i and w.lower() in _CONECTIVAS:
            partes.append(w)
        else:
            partes.append(w.capitalize())
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


def _con_articulo(nombre: str) -> str:
    """«Primera Sala Civil…» → «la Primera Sala Civil…».

    Sin esto el resolutivo dice «reclamó de Primera Sala Civil», que no es
    español. El artículo se elige por la primera palabra, y si ya viene con él
    no se duplica.
    """
    n = _normalizar_autoridad(nombre)
    if not n:
        return ""
    if re.match(r"^(?:el|la|los|las)\s", n, re.I):
        return n
    primera = n.split()[0].lower()
    femeninas = ("sala", "junta", "primera", "segunda", "tercera", "cuarta",
                 "quinta", "sexta", "séptima", "octava", "novena", "décima",
                 "autoridad", "comisión", "procuraduría", "secretaría",
                 "dirección", "delegación", "subdelegación")
    return f"{'la' if primera in femeninas else 'el'} {n}"


HUECO = "*********"

_PLURAL = {"fundado": "fundados", "infundado": "infundados",
           "inoperante": "inoperantes", "ineficaz": "ineficaces"}


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
    if len(unicos) == 2:
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


def escribir_precepto(doc, texto_articulo: str, ley: str, num: str):
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
    # EL ACERVO GUARDA UNA MIGAJA DELANTE: «[Ley de Amparo | CAPÍTULO X
    # Sentencias | Disposiciones Fundamentales] Artículo 79. La autoridad…».
    # Es su índice interno, no el precepto, y transcrita queda ridícula en una
    # sentencia. Se quita, y con ella el «Artículo N.» duplicado que viene
    # detrás.
    cuerpo = re.sub(r"^\s*\[[^\]]{0,200}\]\s*", "", cuerpo)
    for _ in range(2):
        cuerpo = re.sub(r"^\s*ART[ÍI]CULO\s+\d+[^.]{0,14}\.?[-–]?\s*", "",
                        cuerpo, flags=re.I)
    q = doc.add_paragraph()
    r = q.add_run(f"«Artículo {num}. {cuerpo}»")
    _fmt(q, sangria=False, tamano=TAMANO_CITA,
         interlineado=INTERLINEADO_CITA)
    q.paragraph_format.left_indent = SANGRIA_CITA
    q.paragraph_format.right_indent = Cm(0.5)
    q.paragraph_format.space_before = Pt(6)
    q.paragraph_format.space_after = Pt(6)
    q.paragraph_format.keep_with_next = False
    return q


def _preceptos_del_parrafo(texto: str, normas: list) -> list:
    """[(número, norma)] de los artículos que este párrafo cita y tenemos."""
    fuera, vistos = [], set()
    for m in _RX_ARTICULO_CITADO.finditer(texto or ""):
        num = m.group(1)
        if num in vistos:
            continue
        n = _norma_del_texto(m.group(0), num, normas)
        if n and str(n.get("texto") or "").strip():
            vistos.add(num)
            fuera.append((num, n))
    return fuera


def notas_de_articulos(doc, p, texto: str, normas: list, notas: list) -> int:
    """Cuelga del párrafo el texto de los artículos que cita. Devuelve cuántos."""
    if not normas:
        return 0
    puestos = 0
    for m in _RX_ARTICULO_CITADO.finditer(texto or ""):
        if puestos >= MAX_ARTICULOS_POR_PARRAFO:
            break
        num = m.group(1)
        n = _norma_del_texto(m.group(0), num, normas)
        if not n:
            continue
        cuerpo = " ".join(str(n.get("texto", "")).split())[:900]
        if not cuerpo:
            continue
        _ley = n.get("cuerpo_legal") or n.get("fuente") or ""
        # El texto del acervo ya suele venir con «Artículo N.» delante.
        cuerpo = re.sub(r"^\s*\[[^\]]{0,200}\]\s*", "", cuerpo)
        cuerpo = re.sub(r"^\s*ART[ÍI]CULO\s+\d+[^.]{0,12}\.?\s*", "", cuerpo,
                        flags=re.I)
        pie = f"«Artículo {num}. {cuerpo}» — {_ley}".strip()
        if pie in notas:
            continue
        notas.append(pie)
        _run_llamada(p, len(notas))
        puestos += 1
    return puestos


MAX_ARTICULOS_POR_PARRAFO = 1
MAX_NOTAS_DE_ARTICULOS = 8


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


def _sin_eco(texto: str, cuerpo_tesis: str) -> str:
    """Quita del párrafo las frases que repiten la tesis ya transcrita.

    Pedirlo en el prompt no basta: se dijo en el cuerpo y al final, y aun así
    el modelo vuelve a contar la tesis en una de cada dos corridas. Se hace
    aquí, que es donde no falla, y QUIRÚRGICAMENTE: se borran las frases que
    repiten y se conserva lo que aplica el criterio al caso, que es lo único
    que el lector no tiene ya delante.
    """
    if not (cuerpo_tesis or "").strip():
        return texto
    voc = [set(_norm_palabras(f)) for f in _frases_de(cuerpo_tesis)]
    if not voc:
        return texto
    quedan = []
    for frase in re.split(r"(?<=[.])\s+", texto or ""):
        p = set(_norm_palabras(frase))
        if len(p) >= 8 and any(len(p & v) / max(1, len(p)) > 0.72 for v in voc):
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
    # LA FRASE QUE INTRODUCÍA EL EXTRACTO SE QUEDA COJA. Al borrar el
    # entrecomillado, «…, que dispone: «…».» se convierte en «…, que dispone.»,
    # que es peor que la repetición: no dice nada y se nota. Se poda el
    # introductor entero, esté al final o en medio del párrafo, y la frase se
    # cierra donde acababa el sujeto.
    fuera = re.sub(
        r"[,;]?\s*(?:en\s+la\s+parte\s+conducente[,\s]*)?"
        r"(?:el\s+cual|la\s+cual|que|y\s+que|donde)?\s*"
        r"(?:dispone|establece|se[ñn]ala|prev[ée]|dice|reza|indica|prescribe)"
        r"(?:\s+lo\s+siguiente)?\s*:?\s*(?=[.;]|$)",
        "", fuera, flags=re.I)
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
_RX_INICIO_EFECTOS = re.compile(
    r"(?:^|\s)(?:por\s+tanto,?\s+)?la\s+concesi[óo]n\s+del\s+amparo\s+debe\s+"
    r"producir\s+los\s+efectos|^\s*los\s+efectos\s+de\s+la\s+concesi[óo]n|"
    r"^\s*el\s+primer\s+efecto\s+consiste", re.I)
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
    # El párrafo final de cierre —«por lo que procede conceder el amparo…»— no
    # es un efecto: cierra el estudio y ya lo pone el resolutivo.
    while efectos and re.search(r"procede\s+conceder\s+el\s+amparo",
                                efectos[-1], re.I) and len(efectos) > 1:
        efectos.pop()
    return cuerpo, efectos


def _escribir_estudio(doc, estudio, tesis, notas, normas=None) -> int:
    """Los párrafos del estudio, con sus citas rehechas desde el acervo."""
    citadas = 0
    ultima_tesis = None
    transcritos = set()
    transcritas_tesis = set()
    for t in (estudio or []):
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
        hallada, m_r = tesis_del_rubro(t, tesis or [])
        # Y UNA TESIS TAMBIÉN. La 169606 se transcribió dos veces en el mismo
        # considerando —una en el marco y otra al contestar el concepto—: el
        # texto íntegro repetido no aporta y alarga la sentencia sin decir nada.
        if hallada and str(hallada.get("registro") or "") in transcritas_tesis:
            hallada = None
        if hallada and m_r and citadas < MAX_CITAS_DOCUMENTO:
            transcritas_tesis.add(str(hallada.get("registro") or ""))
            antes = _RX_COLA_ANUNCIO.sub("", t[:m_r.start()].rstrip(" ,;:"))
            cola = t[m_r.end():].lstrip(" ,;:.")
            escribir_cita(doc, hallada, antes.rstrip(" ,;:"), notas)
            citadas += 1
            cola = _sin_eco(cola, hallada.get("texto") or "")
            if len(cola.split()) > 6:
                parrafo_con_citas(doc, cola, notas)
            ultima_tesis = hallada
            continue
        if ultima_tesis is not None:
            t = _sin_eco(t, ultima_tesis.get("texto") or "")
            ultima_tesis = None
            if len(t.split()) < 6:
                continue
        # Se decide ANTES de escribir qué artículos van a transcribirse, para
        # poder quitar del párrafo el extracto que quedaría repetido debajo.
        _preceptos = [(n_, x) for n_, x in
                      _preceptos_del_parrafo(t, normas)[:MAX_ARTICULOS_POR_PARRAFO]
                      if n_ not in transcritos]
        t = _sin_extracto_repetido(t, _preceptos)
        if len(t.split()) < 6:
            continue
        p_ = parrafo_con_citas(doc, t, notas)
        # EL PRECEPTO SE TRANSCRIBE, NO SE RESUME. Va en bloque aparte, con
        # sangría y a un espacio, detrás del párrafo que lo anuncia.
        if p_ is not None:
            for num, n_ in _preceptos:
                # UN ARTÍCULO SE TRANSCRIBE UNA VEZ. La clave era (número,
                # ley) y el 48 salió DOS veces porque llegó por dos caminos con
                # el nombre de la ley escrito distinto —«Artículo 48.-» y
                # «Artículo 48.»—. Al lector le da igual de dónde vino: lee lo
                # mismo dos veces seguidas.
                if num in transcritos:
                    continue
                transcritos.add(num)
                escribir_precepto(doc, n_.get("texto"),
                                  n_.get("cuerpo_legal") or n_.get("fuente") or "",
                                  num)
    return citadas


# ═══ EL ESQUELETO DE CADA TIPO, MEDIDO ═════════════════════════════════════
# amparo directo civil 26 · administrativo 16 · queja 20 (y 153 de recuento) ·
# revisión civil 31 · administrativa 16 · fiscal 28. La regla que vale para
# TODOS: el resumen de lo recurrido NO es considerando; el considerando que
# lleva su nombre es la DISPENSA de transcribirlo.
#
# Lo que cambia de un tipo a otro y rompería una plantilla única:
#   · Los RECURSOS no tienen «Existencia del acto reclamado»: es del amparo.
#   · En la QUEJA el cómputo va en PROSA, sin tabla, y la procedencia lleva
#     UNA nota al pie con el artículo 97 de la Ley de Amparo.
#   · El secretario escribe «Trascripción» sin la n: 104 veces contra 12.
#   · El estudio no tiene ordinal fijo: es el último, y su número sale de
#     cuántos apartados le preceden.
ESQUELETO = {
    # ═══════════════════════════════════════════════════════════════════
    # LA TABLA DE CÓMPUTO CASI NUNCA SE DIBUJA, Y YO LA DIBUJABA SIEMPRE
    # ═══════════════════════════════════════════════════════════════════
    # Contado sobre los adelantos reales del corpus:
    #
    #   amparo directo      tabla en  1 de 45 · «oportuna a la luz del art.» 22
    #   amparo en revisión  tabla en  1 de 45 · «oportuna a la luz del art.» 29
    #   revisión fiscal     tabla en 13 de 29
    #   queja               tabla en  0 de 21 · «oportuna a la luz del art.» 16
    #
    # El secretario DECLARA la oportunidad citando el precepto —«la
    # presentación de la demanda resultó oportuna, a la luz del artículo 17 de
    # la Ley de Amparo»— y sólo desglosa el cómputo en la revisión fiscal, y ni
    # siquiera siempre. Yo lo hacía al revés: tabla en tres de los cuatro tipos.
    #
    # El cómputo SE SIGUE HACIENDO —es lo que detecta una extemporaneidad y
    # para el pipeline— pero se ESCRIBE como él lo escribe.
    "amparo_directo": {
        "q": "conceptos de violación",
        "recurrido": "la sentencia reclamada",
        "tabla_computo": False,
        "dispensa": "Acto reclamado y {q}.",
        "legitimacion": "Legitimación y oportunidad.",
        "existencia": True,
        "sub_recurrido": "Sentencia reclamada",
        "adhesivo": "Amparo adhesivo.",
    },
    "amparo_revision": {
        "q": "agravios",
        "recurrido": "la resolución recurrida",
        "tabla_computo": False,
        "dispensa": "Resolución recurrida y {q} de la parte recurrente.",
        "legitimacion": "Legitimación y oportunidad para interponer el recurso.",
        "existencia": False,
        "sub_recurrido": "Resolución recurrida",
        "adhesivo": "Revisión adhesiva.",
    },
    "queja": {
        "q": "agravios",
        "recurrido": "el auto recurrido",
        # EN LA QUEJA EL CÓMPUTO VA EN PROSA. Medido: ni una tabla en 20
        # documentos. Dibujarla aquí sería inventarle un formato al secretario.
        "tabla_computo": False,
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
    return s


def _encabezado(doc, texto):
    for s in doc.sections:
        for parte in (s.header, s.first_page_header, s.even_page_header):
            if parte is None:
                continue
            p = parte.paragraphs[0] if parte.paragraphs else parte.add_paragraph()
            p.text = ""
            r = p.add_run(texto)
            r.bold = True
            r.font.name = FUENTE
            r.font.size = Pt(11)
            r.font.color.rgb = RGBColor.from_string(GRIS_CABECERA)
            p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT


def _caratula(doc, datos):
    """La ficha de identificación. Del asunto, no de ningún otro."""
    campos = [("EXPEDIENTE", datos.get("encabezado", "")),
              ("QUEJOSO", datos.get("quejoso", "")),
              ("AUTORIDAD RESPONSABLE", datos.get("responsable", "")),
              ("MAGISTRADO PONENTE", datos.get("magistrado", "")),
              ("SECRETARIA/O", datos.get("secretario", ""))]
    # LOS DATOS DE IDENTIFICACIÓN VAN A LA IZQUIERDA Y EN MAYÚSCULAS, como en
    # sus proyectos. Justificados y en caja mixta parecían prosa; son una ficha
    # y se leen de un golpe de vista.
    for etiqueta, valor in campos:
        if not valor:
            continue
        p = doc.add_paragraph()
        r1 = p.add_run(f"{etiqueta}: ")
        r1.bold = True
        r2 = p.add_run(str(valor).upper())
        r2.bold = True
        _fmt(p, sangria=False, interlineado=1.0,
             alineacion=WD_ALIGN_PARAGRAPH.LEFT)
        p.paragraph_format.space_after = Pt(0)
    doc.add_paragraph()


def _bloque_firmas(doc, datos):
    for etiqueta, quien in (("MAGISTRADO PONENTE", datos.get("magistrado", "")),
                            ("SECRETARIA/O DE TRIBUNAL", datos.get("secretario", ""))):
        if not quien:
            continue
        parrafo(doc, "", sangria=False)
        parrafo(doc, etiqueta, sangria=False, negrita=True,
                alineacion=WD_ALIGN_PARAGRAPH.CENTER, interlineado=1.0)
        parrafo(doc, str(quien).upper(), sangria=False, negrita=True,
                alineacion=WD_ALIGN_PARAGRAPH.CENTER, interlineado=1.0)


def componer(datos: dict, estructura: Estructura, computo, fecha_en_letra,
             ruta_salida: str, antecedentes=None, resumen_acto=None,
             resumen_conceptos=None, problemas=None, estudio=None,
             calificaciones=None, tesis=None, marco_escrito="",
             tipo_asunto="amparo_directo", normas=None) -> str:
    """Escribe el .docx entero. No hay plantilla de la que partir."""
    doc = docx.Document()
    notas: list = []
    _pagina(doc)
    _encabezado(doc, datos.get("encabezado", ""))
    _caratula(doc, datos)

    if estructura.apertura:
        parrafo(doc, estructura.apertura, sangria=True)
    if estructura.visto:
        # El rótulo lo pone la composición; el modelo lo repite igual aunque se
        # le pida que no —«V I S T O, VISTO, para resolver…»—. Se le quita.
        _v = re.sub(r"^\s*V\s*I\s*S\s*T\s*O\s*S?\s*,?\s*", "",
                    estructura.visto.strip(), flags=re.I)
        tramos(doc, [("V I S T O, ", {"bold": True}), (_v, {})], sangria=True)

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
            p.paragraph_format.keep_with_next = True
            escribir_cuerpo(p)

    def _texto_en(p, texto, resto=None):
        """El primer párrafo continúa el rótulo; el resto van aparte."""
        if (texto or "").strip():
            r = p.add_run(texto.strip())
            r.font.name = FUENTE
            r.font.size = TAMANO
        for x in (resto or []):
            if x.strip():
                parrafo(doc, x.strip())

    # ── RESULTANDO ──
    rotulo(doc, "Resultando")
    res_apartados = []
    for res in (estructura.resultandos or []):
        cuerpo = (res.get("texto") or "").strip()
        if not cuerpo:
            continue
        rot = (res.get("titulo") or "").strip().rstrip(".") + "."
        res_apartados.append((rot, (lambda c: lambda p: _texto_en(p, c))(cuerpo)))
    # La sesión SIEMPRE cierra el resultando y enlaza con el considerando.
    res_apartados.append((
        "Verificación de la sesión vía remota.",
        lambda p: _texto_en(p, f"El presente asunto se listó para la sesión de "
                               f"{HUECO}, la cual se celebró conforme a las "
                               f"disposiciones aplicables; y,")))
    _emitir(res_apartados)

    # ── CONSIDERANDO ──
    rotulo(doc, "Considerando")
    cs = [str(c or "").strip().lower() for c in (calificaciones or []) if c]
    concede = any(c.startswith("fundad") for c in cs)
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
    _fem = bool(re.match(r"\s*(?:la|las)\s", _resp, re.I) or
                re.search(r"\b(sala|junta|secretar[íi]a|comisi[óo]n|"
                          r"procuradur[íi]a|direcci[óo]n)\b", _resp, re.I))
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
    _datos_bk.setdefault("inciso", "b" if _mat in ("administrativa", "agraria") else "c")
    _datos_bk.setdefault("concordancia", "localizada" if _fem else "localizado")
    # EL ARTÍCULO LO PONE LA PLANTILLA, NO YO. Las fórmulas del banco ya dicen
    # «por el {responsable}» y «dictado por el {responsable}», así que
    # anteponerle el artículo aquí producía «por el el Juez de Distrito». Se
    # entrega el nombre limpio y la plantilla lo enmarca; donde hace falta
    # artículo —el resolutivo, los efectos— se pone en ese sitio.
    _datos_bk["responsable"] = _normalizar_autoridad(_resp)
    # Y LOS DATOS QUE LA PLANTILLA PIDE Y NADIE LLENABA. `{descripcion_acto}`
    # salía como hueco «*********» en la competencia de toda queja: es la única
    # frase que dice CONTRA QUÉ se recurre, y sin ella el considerando primero
    # no se sostiene. Sale del propio acto, que el secretario ya subió.
    _datos_bk.setdefault("descripcion_acto", _descripcion_del_acto(datos, tipo_asunto))
    # LA FRACCIÓN DEL ACUERDO GENERAL ES LA DE CADA CIRCUITO. El banco traía
    # escrita la XXII, que es la que reparte la jurisdicción del Vigésimo
    # Segundo: el considerando PRIMERO de un secretario de Mérida nombraba bien
    # a su tribunal y fundaba su competencia en la fracción de otro. No se puede
    # deducir del expediente, así que se deja HUECO VISIBLE, que es la regla de
    # la casa: un hueco se ve y se rellena; una fracción equivocada se firma.
    # LOS MARCADORES DE LA PLANTILLA DE REVISIÓN, que nadie llenaba: salían como
    # «en materia *********, por el *********» en mitad del considerando de
    # competencia. El juez de distrito ES la autoridad responsable —en una
    # revisión de amparo indirecto lo recurrido es su sentencia— y la materia ya
    # está calculada tres líneas más arriba.
    # CON SU ARTÍCULO, porque la plantilla ya no lo pone: «por el Sala Regional»
    # no es español y la concordancia depende del órgano, no de la frase.
    _datos_bk.setdefault("juez_distrito", _con_articulo(_resp) or HUECO)
    _datos_bk.setdefault("juzgado", _con_articulo(_resp) or HUECO)
    _datos_bk.setdefault("recurrente", str(datos.get("quejoso") or "").strip() or HUECO)
    _datos_bk.setdefault("fraccion_acuerdo",
                         str(datos.get("fraccion_acuerdo") or "").strip() or HUECO)
    _datos_bk.setdefault("fecha_acto", str(datos.get("fecha_acto") or "").strip()
                         or HUECO)
    _datos_bk.setdefault("objeto", (f"una sentencia definitiva en materia {_mat}"
                                    if _mat else "una sentencia definitiva"))
    _huecos_bk = []

    def _del_banco(ident, respaldo):
        """La frase del oficio si el banco la tiene; si no, la del modelo."""
        t, faltan = _bk.texto_de(tipo_asunto, ident, _datos_bk)
        if t:
            _huecos_bk.extend(faltan)
            return t
        return respaldo


    _comp = _del_banco("competencia", estructura.competencia)
    if (_comp or "").strip():
        con_apartados.append((_bk.rotulo_de(tipo_asunto, "competencia", "Competencia."),
                              (lambda c: lambda p: _texto_en(p, c))(_comp)))
    _exi = _del_banco("existencia", estructura.existencia)
    if esq["existencia"] and (_exi or "").strip():
        con_apartados.append((_bk.rotulo_de(tipo_asunto, "existencia",
                                            "Existencia del acto reclamado."),
                              (lambda c: lambda p: _texto_en(p, c))(_exi)))

    # Legitimación y oportunidad, con LA TABLA detrás.
    from fase0_oportunidad import parrafo_oportunidad
    import tipos_asunto as _ta
    # EL FUNDAMENTO Y EL VOCABULARIO SALEN DEL CATÁLOGO. Antes decía siempre
    # «el precepto 17 del mencionado ordenamiento», que es el del amparo: una
    # queja se fundaba en el artículo del amparo y una revisión fiscal, en la
    # Ley de Amparo cuando la suya es la LFPCA.
    _op = parrafo_oportunidad(
        computo,
        _ta.plazo_de(tipo_asunto, "").get("fundamento") or "artículo 17 de la Ley de Amparo",
        tipo_asunto)

    def _legitimacion(p):
        _texto_en(p, _op)
        # En la QUEJA el cómputo va en prosa: ni una tabla en 20 documentos.
        if esq["tabla_computo"]:
            tabla_computo(doc, computo, fecha_en_letra)

    con_apartados.append((_bk.rotulo_de(tipo_asunto, "legitimacion",
                                        esq["legitimacion"]), _legitimacion))

    # PROCEDENCIA NO ES UN CONSIDERANDO PROPIO cuando hay «Existencia del acto
    # reclamado»: medido, es su ALTERNATIVA —3 de 26, en asuntos venidos de
    # juez y no de sala—, no un apartado más. Emitirlos los dos corría todo un
    # ordinal y dejaba el Estudio en SÉPTIMO donde el corpus lo tiene SEXTO.
    # PROCEDENCIA sólo donde el corpus la tiene como apartado propio —queja
    # (14 de 20) y revisión fiscal (16 de 28)— o en el amparo cuando sustituye
    # a «Existencia del acto reclamado». En revisión civil aparece en 2 de 31:
    # emitirla por defecto ahí corría un ordinal contra la medida.
    if (estructura.procedencia or "").strip() and (
            esq.get("procedencia_propia")
            or (esq["existencia"] and not (estructura.existencia or "").strip())):
        con_apartados.append(("Procedencia.",
                              (lambda c: lambda p: _texto_en(p, c))(estructura.procedencia)))

    # LA DISPENSA. El rótulo promete el acto reclamado y los conceptos, y el
    # contenido es justamente que NO hace falta transcribirlos: 21 de 26.
    def _dispensa(p):
        _texto_en(
            p,
            _del_banco("dispensa", "") or
            f"Es innecesario transcribir el contenido de "
            f"{esq['recurrido']} y los {q} hechos valer, pues el deber formal "
            f"y material de "
            f"exponer los argumentos legales que sustenten esta resolución no "
            f"depende de la reproducción literal de los aspectos que conforman "
            f"la litis, sino de su adecuado análisis.")

    con_apartados.append((esq["dispensa"].format(q=q), _dispensa))

    if antecedentes:
        def _antecedentes(p):
            _texto_en(p,
                      "Previo al análisis de los planteamientos que se proponen, "
                      "es menester relatar los hechos relevantes del asunto.")
            for x in (antecedentes or []):
                if x.strip():
                    parrafo_con_citas(doc, x.strip(), notas)
        con_apartados.append(("Antecedentes.", _antecedentes))

    # EL ESTUDIO. Es el ÚLTIMO considerando salvo que detrás vaya Efectos, y
    # lleva dentro los subtítulos en negrita, sin ordinal.
    _cuerpo_estudio, _efectos_escritos = partir_efectos(estudio or [])

    def _estudio(p):
        calif = _calificacion_plural(cs)
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
        _escribir_estudio(doc, _cuerpo_estudio, tesis, notas, normas)
        if not concede:
            parrafo(doc,
                    f"Por lo expuesto, dado lo {_calificacion_plural(cs) or 'infundado'} "
                    f"de los {q}, lo procedente es negar el amparo solicitado.")

    con_apartados.append(("Estudio.", _estudio))

    if concede:
        def _efectos(p):
            if _efectos_escritos:
                _texto_en(p, _efectos_escritos[0], _efectos_escritos[1:])
                return
            _texto_en(p,
                      f"Consecuentemente, procede conceder el amparo y protección "
                      f"de la Justicia Federal a {datos.get('quejoso','')} para el "
                      f"efecto de que "
                      f"{_con_articulo(datos.get('responsable','')) or HUECO} "
                      f"deje insubsistente la sentencia reclamada y dicte otra en "
                      f"la que atienda los lineamientos de esta ejecutoria.")
        con_apartados.append(("Efectos.", _efectos))

    _emitir(con_apartados)

    # ── RESUELVE ──
    p_pe = parrafo(doc, "Por lo expuesto y fundado, se:", sangria=True)
    p_pe.paragraph_format.space_before = Pt(14)
    rotulo(doc, "Resuelve")
    _res = RESOLUTIVO.get(str(tipo_asunto or "").strip().lower(),
                          RESOLUTIVO["amparo_directo"])
    if _res.get("punto"):
        # Queja y revisión: NO amparan. Se califica el recurso o se resuelve
        # sobre la sentencia recurrida, que es lo que hacen los engroses.
        _cal = _res["calif"][0] if concede else _res["calif"][1]
        _texto = _res["punto"].format(
            calificacion=_cal,
            responsable=_con_articulo(datos.get("responsable", "")) or HUECO)
        _texto = _texto.replace("por el ", "por el ").replace(" de el ", " del ")
        tramos(doc, [("ÚNICO. ", {"bold": True}), (_texto, {})], sangria=False)
    else:
        formula = _AMPARA if concede else _NO_AMPARA
        tramos(doc, [("ÚNICO. ", {"bold": True}),
                     ("La Justicia de la Unión ", {}),
                     (formula, {"bold": True}),
                     (f" a {datos.get('quejoso','') or HUECO}, en contra de "
                      f"{esq.get('recurrido','la sentencia reclamada')}, dictada "
                      f"por {_con_articulo(datos.get('responsable','')) or HUECO}, "
                      f"precisada en el primer resultando de esta ejecutoria.", {})],
               sangria=False)

    parrafo(doc, _res["notif"], sangria=True)

    _bloque_firmas(doc, datos)
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
    doc.save(ruta_salida)
    _inyectar_notas(ruta_salida, notas)
    return ruta_salida
