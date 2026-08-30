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
             calificaciones=None, tesis=None) -> str:
    """Escribe el .docx entero. No hay plantilla de la que partir."""
    doc = docx.Document()
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

    # ── EL ESTUDIO ──
    for i, t in enumerate(estudio or []):
        if not t.strip():
            continue
        # El estudio ya trae su propio encabezado ordinal desde fase 6.
        parrafo(doc, t.strip())

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
    return ruta_salida
