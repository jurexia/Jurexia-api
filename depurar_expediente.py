"""DEPURAR EL ESCANEO ANTES DE LEERLO.

Del Expediente Electrónico no llega un documento: llega un TOMO. El escaneo de
la «Promoción 1» del 91/2025 son 117 páginas y 345.944 caracteres, y dentro
viven cosas distintas que el pipeline recibía pegadas en un solo muro de texto:

    págs   1-2    portada de la Oficina de Correspondencia Común
    págs   3-4    oficio de remisión y acuse de recibo
    págs   5-84   EL ESCRITO DE AGRAVIOS DEL SAT          ← 80 páginas
    págs  85-94   portadas y sellos de la Sala
    págs  95-112  LA SENTENCIA RECURRIDA DEL TFJA         ← 18 páginas
    págs 113-117  evidencia criptográfica de la firma

El clasificador miraba el conjunto y decía «sentencia_recurrida», porque es lo
que más se le parecía. Los agravios —lo que hay que contestar— nunca llegaban
identificados como agravios.

UNA EQUIVOCACIÓN QUE CASI CUESTA EL DOCUMENTO. Al medir por primera vez, las
79 páginas del bloque grande empezaban todas por «Entrega Personal · Hacienda ·
SAT», y las di por acuses de entrega. No lo son: eso es el MEMBRETE del escrito
del SAT, impreso en cada hoja. Un filtro construido sobre aquella lectura habría
tirado a la basura el documento más importante del expediente. De ahí que aquí
no se borre nada por su pinta: se SEGMENTA y se etiqueta, y quien decide qué
sobra es el secretario, mirando la lista.

CÓMO SE SEGMENTA, Y POR QUÉ ASÍ

Se probaron tres señales sobre el documento real:

  1. Huella del encabezado (primeros 260 caracteres). Los agravios salían
     perfectos; la sentencia se partía en 13 trozos, porque está escaneada A
     DOBLE CARA y el membrete sólo va en una.
  2. Líneas de membrete repetidas. Mejor, pero el OCR estropea el escudo de
     formas distintas en cada hoja —«UNIDOS TEJA», «UNIDOS STAD», «OOVEST
     TEJA»— y la semejanza se hundía.
  3. IDENTIFICADORES, que es lo que de verdad nombra a un documento judicial:
     el número de expediente, el de oficio, el RFC. Éstos el OCR los respeta.

Con la tercera hubo que añadir una cosa más. Tomando los identificadores de
toda la página, la sentencia volvía a partirse: cita tantas veces los oficios
que ANALIZA que ésos ganaban en frecuencia al suyo propio. El identificador que
nombra a un documento está en su ENCABEZADO; los que aparecen en el cuerpo son
citas. Mirando sólo los primeros 380 caracteres, el corte sale limpio.
"""

import re
import unicodedata
from collections import Counter

# ── QUÉ NOMBRA A UN DOCUMENTO ─────────────────────────────────────────────────
IDENTIFICADORES = [
    # 695/25-09-01-7-OT · expediente del Tribunal Federal de Justicia Administrativa
    ("exp_tfja", r"\b\d{3,5}/\d{2}-\d{2}-\d{2}-\d(?:-[A-Z]{2})?\b"),
    # 600-47-00-00-00-2025-4947 · oficio de autoridad fiscal
    ("oficio", r"\b\d{3}[ -]\d{2}[ -]\d{2}[ -]\d{2}[ -]\d{2}[ -]20\d{2}[ -]\d{3,4}\b"),
    ("rfc", r"\b[A-Z]{3,4}\d{6}[A-Z0-9]{3}\b"),
    # 91/2025 · expediente de tribunal colegiado. Va el ÚLTIMO a propósito: es
    # el patrón más goloso y engancha fechas y citas de tesis.
    ("exp", r"\b\d{1,4}/20\d{2}\b"),
]

ZONA_ENCABEZADO = 380      # caracteres desde el principio de la página
MIN_PAGINAS_DOC = 3        # menos que esto es portada, sello o acuse


def _normalizar(linea: str) -> str:
    t = unicodedata.normalize("NFKD", linea.upper())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z ]", " ", t)).strip()


def _identificadores(texto: str) -> set:
    s = set()
    for nombre, rx in IDENTIFICADORES:
        for m in re.findall(rx, texto):
            s.add(f"{nombre}:{re.sub(r'[ -]', '', m)}")
    return s


def membrete(paginas, umbral=0.06):
    """Las líneas que se repiten página tras página: membrete, sello, pie.

    No son contenido. En el 91/2025 son el 9,3% del texto, y su daño no es el
    tamaño sino que el modelo las lee como si fueran argumento.
    """
    n = len(paginas)
    if n < 4:
        return set()
    corte = max(4, int(n * umbral))
    cuenta = Counter()
    for p in paginas:
        arriba = [_normalizar(x) for x in p.split("\n")][:14]
        cuenta.update({l for l in arriba if len(l) >= 8})
    return {l for l, c in cuenta.items() if c >= corte}


def sin_membrete(texto: str, lineas_membrete: set) -> str:
    if not lineas_membrete:
        return texto
    fuera = []
    for x in texto.split("\n"):
        n = _normalizar(x)
        # LA CADENA DE FIRMA ELECTRÓNICA. Va en las 117 páginas y no dice nada:
        # es el número de serie del certificado, en hexadecimal.
        if re.fullmatch(r"[0-9a-fA-F ]{20,}", x.strip()):
            continue
        if n and n in lineas_membrete:
            continue
        fuera.append(x)
    return "\n".join(fuera)


def segmentar(paginas):
    """Parte el tomo en documentos. Devuelve [(primera, ultima, etiqueta), ...],
    numeradas desde 1 y con los extremos incluidos."""
    n = len(paginas)
    if not n:
        return []
    cabeceras = [_identificadores(p[:ZONA_ENCABEZADO]) for p in paginas]
    cuenta = Counter()
    for s in cabeceras:
        cuenta.update(s)

    # De los identificadores del encabezado, el que más veces sale en TODO el
    # tomo: ése es el del documento. Los que salen una vez son citas.
    etiquetas = [max(s, key=lambda k: cuenta[k]) if s else None for s in cabeceras]
    # Una cara sin membrete continúa el documento de la anterior. Ésta es la
    # línea que hace funcionar el escaneo a doble cara.
    for i in range(n):
        if etiquetas[i] is None and i:
            etiquetas[i] = etiquetas[i - 1]

    bloques, inicio = [], 0
    for i in range(1, n):
        if etiquetas[i] != etiquetas[i - 1]:
            bloques.append((inicio + 1, i, etiquetas[i - 1]))
            inicio = i
    bloques.append((inicio + 1, n, etiquetas[-1]))

    # Fundir lo contiguo con la misma etiqueta, y absorber los trocitos sueltos
    # —una portada, un sello— en el documento que los precede.
    fundidos = []
    for a, b, e in bloques:
        if fundidos and (e == fundidos[-1][2]
                         or ((b - a + 1) < MIN_PAGINAS_DOC and cuenta.get(e, 0) < 5)):
            fundidos[-1] = (fundidos[-1][0], b, fundidos[-1][2])
        else:
            fundidos.append((a, b, e))
    return fundidos


def depurar(paginas):
    """El tomo, hecho documentos limpios.

    Cada uno trae sus páginas, su texto ya sin membrete ni cadenas de firma, y
    el peso que tiene dentro del tomo. Clasificarlo es cosa de `fase_sise`, que
    ya sabe distinguir una sentencia de un auto de admisión leyéndolo.
    """
    lineas_membrete = membrete(paginas)
    salida = []
    for a, b, etiqueta in segmentar(paginas):
        trozos = [sin_membrete(p, lineas_membrete) for p in paginas[a - 1:b]]
        texto = "\n\n".join(t for t in trozos if t.strip())
        salida.append({
            "desde": a, "hasta": b, "paginas": b - a + 1,
            "etiqueta": etiqueta or "",
            "texto": texto,
            "caracteres": len(texto),
        })
    return salida
