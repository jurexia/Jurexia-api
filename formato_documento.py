"""LO QUE EL ARCHIVO ES, NO LO QUE DICE SU NOMBRE, Y EL WORD HECHO DE FOTOS.   (29-oct-2026)

«No se pudo extraer texto» le salió a 9 usuarios de pago en 22 análisis de 30
días, y no era un archivo malo:

  · 10 eran archivos con el nombre cambiado: un .docx que por dentro es un
    .doc viejo, RTF o HTML («File is not a zip file», ocho veces al mismo
    abogado Pro con sus contratos) y un .doc que por dentro es otra cosa
    («not an OLE2 structured storage file»). Word los abre sin quejarse, así
    que el abogado no tiene forma de saberlo. Aquí se mira la firma de los
    bytes y se lee con el lector que corresponde.
  · 11 eran Word sin texto: escaneos o capturas pegados como imágenes
    («imagenescontrato.docx», «CAPTURA DE PANTALLA.docx», «COMPLETO.docx» de
    20 MB). Aquí se sacan sus imágenes en el orden del documento y se arma
    con ellas un PDF, que pasa por el mismo OCR que un PDF escaneado.

Sin red: todo esto es lectura local de bytes (el OCR lo decide quien llama).
"""
from __future__ import annotations

import io
import re
import zipfile
from typing import List, Optional

_OLE = b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1"
_OPC_XML = dict(resolve_entities=False, no_network=True, load_dtd=False, huge_tree=False)
# Lo que PyMuPDF sabe insertar como página. Los EMF/WMF (dibujos de Office) no.
_IMAGEN_OK = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".jfif", ".webp")


def formato_real(contenido: bytes) -> Optional[str]:
    """pdf | docx | odt | doc | rtf | html | texto, o None si no se reconoce.
    Por la firma de los bytes, que es lo que de verdad decide cómo se lee."""
    if not contenido:
        return None
    cabeza = contenido[:2048]
    if b"%PDF-" in cabeza[:1024]:
        return "pdf"
    if contenido[:4] == b"PK\x03\x04":
        try:
            with zipfile.ZipFile(io.BytesIO(contenido)) as z:
                nombres = set(z.namelist())
                if "word/document.xml" in nombres or any(n.endswith("document.xml") and n.startswith("word/") for n in nombres):
                    return "docx"
                if "mimetype" in nombres and b"opendocument.text" in z.read("mimetype")[:100]:
                    return "odt"
                if "[Content_Types].xml" in nombres and b"wordprocessingml" in z.read("[Content_Types].xml"):
                    return "docx"
        except Exception:
            return None
        return None
    if contenido[:8] == _OLE:
        return "doc"
    inicio = cabeza.lstrip(b"\xef\xbb\xbf \t\r\n")
    if inicio[:5].lower() == b"{\\rtf":
        return "rtf"
    minus = inicio[:600].lower()
    if minus.startswith(b"<") and (b"<html" in minus or b"<!doctype html" in minus or b"<w:worddocument" in minus
                                    or b"urn:schemas-microsoft-com:office" in minus or b"<?xml" in minus):
        return "html"
    # Texto plano: casi todo imprimible en UTF-8 o Latin-1.
    try:
        muestra = cabeza.decode("utf-8")
    except UnicodeDecodeError:
        muestra = cabeza.decode("latin-1", errors="replace")
    if muestra and sum(c.isprintable() or c in "\r\n\t" for c in muestra) / len(muestra) > 0.97:
        return "texto"
    return None


# ── RTF ───────────────────────────────────────────────────────────────────
# Lo justo para el texto: párrafos, tabuladores, \'hh (Windows-1252), \uN y
# fuera los grupos que no son contenido (tablas de fuentes, colores, estilos,
# imágenes, metadatos y todo lo marcado con \*).
_RTF_TOKEN = re.compile(r"\\([a-zA-Z]+)(-?\d+)? ?|\\'([0-9a-fA-F]{2})|\\([^a-zA-Z])|([{}])|([^\\{}]+)", re.S)
_RTF_FUERA = {"fonttbl", "colortbl", "stylesheet", "info", "pict", "object", "header", "footer",
              "headerl", "headerr", "footerl", "footerr", "themedata", "colorschememapping",
              "datastore", "latentstyles", "listtable", "listoverridetable", "rsidtbl", "generator",
              "xmlnstbl", "mmathPr", "pgdsctbl", "filetbl", "revtbl"}


def texto_rtf(contenido: bytes) -> str:
    crudo = contenido.decode("latin-1", errors="replace")
    pila: List[tuple] = []
    ignorar, saltar_uc, uc = False, 0, 1
    salida: List[str] = []
    for m in _RTF_TOKEN.finditer(crudo):
        palabra, arg, hexa, simbolo, llave, texto = m.groups()
        if llave == "{":
            pila.append((ignorar, uc))
            continue
        if llave == "}":
            if pila:
                ignorar, uc = pila.pop()
            continue
        if simbolo is not None:
            if simbolo == "*":
                ignorar = True
            elif not ignorar and simbolo in "\\{}":
                salida.append(simbolo)
            elif not ignorar and simbolo == "~":
                salida.append(" ")
            continue
        if palabra is not None:
            if palabra in _RTF_FUERA:
                ignorar = True
            elif palabra == "uc" and arg:
                uc = int(arg)
            elif ignorar:
                pass
            elif palabra in ("par", "line", "sect", "page", "row"):
                salida.append("\n")
            elif palabra in ("tab", "cell"):
                salida.append("\t")
            elif palabra == "u" and arg:
                n = int(arg)
                salida.append(chr(n + 65536 if n < 0 else n))
                saltar_uc = uc
            continue
        if hexa is not None:
            if saltar_uc:
                saltar_uc -= 1
            elif not ignorar:
                salida.append(bytes([int(hexa, 16)]).decode("cp1252", errors="replace"))
            continue
        if texto is not None and not ignorar:
            t = texto.replace("\r", "").replace("\n", "")
            if saltar_uc and t:
                quitar = min(saltar_uc, len(t))
                t, saltar_uc = t[quitar:], saltar_uc - quitar
            salida.append(t)
    return re.sub(r"\n{3,}", "\n\n", "".join(salida)).strip()


def texto_html(contenido: bytes) -> str:
    from lxml import html as _html
    try:
        arbol = _html.fromstring(contenido)
    except Exception:
        return ""
    for el in arbol.xpath("//script|//style"):
        el.drop_tree()
    for el in arbol.xpath("//br|//p|//div|//tr|//li|//h1|//h2|//h3|//h4"):
        el.tail = "\n" + (el.tail or "")
    return re.sub(r"[ \t]+\n", "\n", re.sub(r"\n{3,}", "\n\n", arbol.text_content())).strip()


def texto_plano(contenido: bytes) -> str:
    try:
        return contenido.decode("utf-8").strip()
    except UnicodeDecodeError:
        return contenido.decode("latin-1", errors="replace").strip()


def texto_odt(contenido: bytes) -> str:
    from lxml import etree
    with zipfile.ZipFile(io.BytesIO(contenido)) as z:
        raiz = etree.fromstring(z.read("content.xml"), etree.XMLParser(**_OPC_XML))
    ns = {"text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0"}
    parrafos = ["".join(p.itertext()) for p in raiz.iterfind(".//text:p", ns)] + \
               ["".join(h.itertext()) for h in raiz.iterfind(".//text:h", ns)]
    return "\n\n".join(p for p in parrafos if p.strip())


# ── El Word hecho de fotos ────────────────────────────────────────────────

def imagenes_docx(contenido: bytes, tope: int = 600) -> List[bytes]:
    """Las imágenes del cuerpo del documento, en el orden en que aparecen
    (las relaciones `r:embed` de document.xml), sin repetir y sólo las que
    PyMuPDF sabe poner en una página. Lo que no se pueda leer, se omite."""
    from lxml import etree
    salida: List[bytes] = []
    try:
        z = zipfile.ZipFile(io.BytesIO(contenido))
    except Exception:
        return salida
    with z:
        nombres = set(z.namelist())
        rels = {}
        try:
            raiz_rels = etree.fromstring(z.read("word/_rels/document.xml.rels"), etree.XMLParser(**_OPC_XML))
            for r in raiz_rels:
                destino = str(r.get("Target") or "")
                if destino.startswith("/"):
                    destino = destino.lstrip("/")
                elif not destino.startswith("word/"):
                    destino = "word/" + destino
                rels[r.get("Id")] = destino
        except Exception:
            pass
        orden: List[str] = []
        try:
            with z.open("word/document.xml") as f:
                for _, el in etree.iterparse(f, events=("end",), **_OPC_XML):
                    for k, v in el.attrib.items():
                        if k.endswith("}embed") or k.endswith("}id") and str(el.tag).endswith("imagedata"):
                            destino = rels.get(v)
                            if destino and destino not in orden:
                                orden.append(destino)
                    el.clear()
        except Exception:
            pass
        if not orden:
            orden = sorted(n for n in nombres if n.startswith("word/media/"))
        for n in orden:
            if len(salida) >= tope:
                break
            if n in nombres and n.lower().endswith(_IMAGEN_OK):
                try:
                    datos = z.read(n)
                except Exception:
                    continue
                if len(datos) > 2_000:    # iconos y viñetas no son páginas
                    salida.append(datos)
    return salida


def pdf_de_imagenes(imagenes: List[bytes]) -> Optional[bytes]:
    """Un PDF con una imagen por página, del tamaño de la imagen, para el OCR."""
    import fitz
    doc = fitz.open()
    try:
        for datos in imagenes:
            try:
                pix = fitz.Pixmap(datos)
                # La página mide la imagen a 150 ppp; la imagen entra tal cual,
                # sin volver a muestrearla: el OCR la lee a su resolución.
                ancho, alto = pix.width * 72 / 150, pix.height * 72 / 150
                pix = None
                pagina = doc.new_page(width=max(ancho, 72), height=max(alto, 72))
                pagina.insert_image(pagina.rect, stream=datos)
            except Exception:
                continue
        if doc.page_count == 0:
            return None
        return doc.tobytes(garbage=3, deflate=True)
    finally:
        doc.close()


def pdf_rasterizado(contenido: bytes, dpi: int = 150, tope_paginas: int = 150) -> Optional[bytes]:
    """El mismo PDF con cada página convertida en imagen. Es el repliegue para
    un PDF que el OCR rechaza por su estructura («400 Bad Request» de Azure y
    «INVALID_ARGUMENT» de Gemini con el mismo archivo de un escáner, 29-oct):
    PyMuPDF sí lo abre y lo dibuja, y la copia en imágenes la lee cualquiera."""
    import fitz
    try:
        src = fitz.open(stream=contenido, filetype="pdf")
    except Exception:
        return None
    try:
        if src.page_count == 0 or src.page_count > tope_paginas:
            return None
        doc = fitz.open()
        try:
            escala = dpi / 72.0
            for pagina in src:
                pix = pagina.get_pixmap(matrix=fitz.Matrix(escala, escala), colorspace=fitz.csGRAY)
                nueva = doc.new_page(width=pagina.rect.width, height=pagina.rect.height)
                nueva.insert_image(nueva.rect, stream=pix.tobytes("jpeg", jpg_quality=80))
            return doc.tobytes(garbage=3, deflate=True)
        finally:
            doc.close()
    finally:
        src.close()
