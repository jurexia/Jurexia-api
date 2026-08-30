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


def escribir_cita(doc, t: dict, anuncio: str, notas: list) -> None:
    """El bloque entero de la cita, con su nota al pie."""
    if anuncio.strip():
        parrafo(doc, anuncio.rstrip(" ,;:") + " de rubro y texto siguientes:")

    # El rubro, solo y en negrita.
    p = doc.add_paragraph()
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
        q.paragraph_format.keep_with_next = True

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
        if loc or reg:
            pie = loc if loc else ""
            if reg and reg not in pie:
                pie = (pie + ", " if pie else "") + f"registro digital {reg}"
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
        filas.append(("Días inhábiles descontados",
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
    doc.add_paragraph()


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


def prompt_estructura(datos: dict) -> str:
    q = "agravios" if datos.get("es_recurso") else "conceptos de violación"
    return f"""Eres el secretario de un Tribunal Colegiado de Circuito y escribes las
partes ESTRUCTURALES de una sentencia de amparo directo. No escribes el estudio
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
       "texto": "<quién promovió, cuándo, ante quién, contra qué acto y qué autoridad>"}},
     {{"titulo": "Derechos humanos cuya violación se alega", "texto": "<…>"}},
     {{"titulo": "Admisión y trámite", "texto": "<…>"}}
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
- EL PRECEPTO LOCAL O SECUNDARIO decisivo SÍ se transcribe, entre comillas y
  con su número al frente.
- LA FUENTE CONVENCIONAL —Convención sobre los Derechos del Niño, Convención
  Americana— y los criterios de la CORTE INTERAMERICANA entran SÓLO si el
  problema los exige. Cuando entran, se dice qué obligación imponen, no que
  existen.
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


def _con_articulo(nombre: str) -> str:
    """«Primera Sala Civil…» → «la Primera Sala Civil…».

    Sin esto el resolutivo dice «reclamó de Primera Sala Civil», que no es
    español. El artículo se elige por la primera palabra, y si ya viene con él
    no se duplica.
    """
    n = (nombre or "").strip()
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
    for etiqueta, valor in campos:
        if not valor:
            continue
        tramos(doc, [(f"{etiqueta}: ", {"bold": True}), (str(valor), {})],
               sangria=False, interlineado=1.15)
    doc.add_paragraph()


def _bloque_firmas(doc, datos):
    doc.add_paragraph()
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
             calificaciones=None, tesis=None, marco_escrito="") -> str:
    """Escribe el .docx entero. No hay plantilla de la que partir."""
    doc = docx.Document()
    notas: list = []
    _pagina(doc)
    _encabezado(doc, datos.get("encabezado", ""))
    _caratula(doc, datos)

    if estructura.apertura:
        parrafo(doc, estructura.apertura, sangria=False)
    if estructura.visto:
        # El rótulo lo pone la composición; el modelo lo repite igual aunque se
        # le pida que no —«V I S T O, VISTO, para resolver…»—. Se le quita.
        _v = re.sub(r"^\s*V\s*I\s*S\s*T\s*O\s*S?\s*,?\s*", "",
                    estructura.visto.strip(), flags=re.I)
        tramos(doc, [("V I S T O, ", {"bold": True}), (_v, {})], sangria=False)

    # ── RESULTANDO ──
    rotulo(doc, "Resultando")
    n = 0
    for res in (estructura.resultandos or []):
        if not (res.get("texto") or "").strip():
            continue
        tramos(doc, [(f"{_ORDINALES[min(n, 9)]}. ", {"bold": True}),
                     (f"{res.get('titulo','').strip()}. ", {"bold": True}),
                     (res["texto"].strip(), {})])
        n += 1
    for t in (antecedentes or []):
        if t.strip():
            parrafo(doc, t.strip())

    # ── CONSIDERANDO ──
    rotulo(doc, "Considerando")
    n = 0

    def _apartado(titulo, cuerpo):
        nonlocal n
        if not (cuerpo or "").strip():
            return
        tramos(doc, [(f"{_ORDINALES[min(n, 9)]}. ", {"bold": True}),
                     (f"{titulo}. ", {"bold": True}), (cuerpo.strip(), {})])
        n += 1

    _apartado("Competencia", estructura.competencia)
    _apartado("Existencia del acto reclamado", estructura.existencia)

    # Legitimación y oportunidad, con LA TABLA
    from fase0_oportunidad import parrafo_oportunidad
    _apartado("Legitimación y oportunidad", parrafo_oportunidad(computo))
    tabla_computo(doc, computo, fecha_en_letra)

    _apartado("Procedencia", estructura.procedencia)

    if resumen_acto:
        tramos(doc, [(f"{_ORDINALES[min(n, 9)]}. ", {"bold": True}),
                     ("Consideraciones de la sentencia reclamada. ", {"bold": True}),
                     ((resumen_acto[0] or "").strip(), {})])
        n += 1
        for t in resumen_acto[1:]:
            if t.strip():
                parrafo(doc, t.strip())

    if resumen_conceptos:
        tramos(doc, [(f"{_ORDINALES[min(n, 9)]}. ", {"bold": True}),
                     ("Planteamientos de la parte quejosa. ", {"bold": True}),
                     ((resumen_conceptos[0] or "").strip(), {})])
        n += 1
        for t in resumen_conceptos[1:]:
            if t.strip():
                parrafo(doc, t.strip())

    # ── EL MARCO JURÍDICO, si el asunto lo pidió ──
    if (marco_escrito or "").strip():
        trozos = [x.strip() for x in re.split(r"\n\s*\n", marco_escrito)
                  if x.strip()]
        tramos(doc, [(f"{_ORDINALES[min(n, 9)]}. ", {"bold": True}),
                     ("Marco jurídico aplicable. ", {"bold": True}),
                     (trozos[0], {})])
        n += 1
        for x in trozos[1:]:
            parrafo(doc, x)

    # ── EL ESTUDIO, con sus citas rehechas desde el acervo ──
    # El modelo escribe «…de rubro y texto siguientes: «RUBRO.» La responsable…»
    # y deja la cita partida, sin transcripción y sin localización. Aquí se
    # reconstruye: anuncio, rubro solo en negrita, texto íntegro en cursiva
    # tomado del ACERVO —no de la memoria del modelo— y la localización al pie.
    citadas = 0
    for t in (estudio or []):
        t = (t or "").strip()
        if not t:
            continue
        hallada, m_r = tesis_del_rubro(t, tesis or [])
        if hallada and m_r and citadas < MAX_CITAS_DOCUMENTO:
            antes = _RX_COLA_ANUNCIO.sub("", t[:m_r.start()].rstrip(" ,;:"))
            cola = t[m_r.end():].lstrip(" ,;:.")
            escribir_cita(doc, hallada, antes.rstrip(" ,;:"), notas)
            citadas += 1
            if len(cola.split()) > 6:
                parrafo(doc, cola)
            continue
        parrafo(doc, t)

    # ── RESUELVE ──
    rotulo(doc, "Resuelve")
    cs = [str(c or "").strip().lower() for c in (calificaciones or []) if c]
    concede = any(c.startswith("fundad") for c in cs)
    formula = _AMPARA if concede else _NO_AMPARA
    tramos(doc, [("ÚNICO. ", {"bold": True}),
                 ("La Justicia de la Unión ", {}),
                 (formula, {"bold": True}),
                 (f" a {datos.get('quejoso','')}, contra el acto que reclamó "
                  f"de {_con_articulo(datos.get('responsable',''))}, precisado "
                  f"en el {_ORDINALES[0].lower()} resultando de esta "
                  f"ejecutoria.", {})],
           sangria=False)

    parrafo(doc, "Notifíquese; con testimonio de esta resolución, devuélvanse "
                 "los autos a su lugar de origen y, en su oportunidad, "
                 "archívese el expediente como asunto concluido.", sangria=False)

    _bloque_firmas(doc, datos)
    doc.save(ruta_salida)
    _inyectar_notas(ruta_salida, notas)
    return ruta_salida
