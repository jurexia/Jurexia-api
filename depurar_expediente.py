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
# CUÁNTO TIENE QUE MANDAR UNA ETIQUETA para que abra documento. Medido: el
# oficio del SAT encabeza 79 de 117 páginas y la Sala 11; en cambio, en un
# escrito de amparo de 161 páginas ningún identificador pasa de un puñado.
MIN_MANDO_PAGINAS = 5
MIN_MANDO_FRACCION = 0.07


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

    # ── SÓLO SE CORTA CON PRUEBA ──────────────────────────────────────────
    # Esto se descubrió rompiéndolo. Sobre la demanda de amparo del ADA
    # 203-2025 —161 páginas de UN SOLO escrito— la regla de arriba producía
    # VEINTICUATRO documentos y repartía como «conceptos» las páginas 133-153.
    # Destrozaba un documento sano, que es peor que no tocarlo.
    #
    # La diferencia con el tomo del 91/2025 es medible: allí el oficio del SAT
    # encabeza 79 páginas seguidas y la Sala otras once; aquí ningún
    # identificador manda en más de un puñado, porque un escrito de amparo cita
    # decenas de expedientes y ninguno es el suyo.
    #
    # Así que una etiqueta sólo abre documento si MANDA de verdad. Si ninguna
    # lo hace, el tomo es un documento y se devuelve entero.
    _minimo = max(MIN_MANDO_PAGINAS, int(n * MIN_MANDO_FRACCION))
    _mando = {}
    for e in etiquetas:
        if e:
            _mando[e] = _mando.get(e, 0) + 1
    _fuertes = {e for e, c in _mando.items() if c >= _minimo}
    if not _fuertes:
        return [(1, n, etiquetas[0] if etiquetas else None)]
    etiquetas = [e if e in _fuertes else None for e in etiquetas]
    # Y lo que no lleva etiqueta fuerte continúa el documento anterior.
    for i in range(n):
        if etiquetas[i] is None and i:
            etiquetas[i] = etiquetas[i - 1]
    for i in range(n - 1, -1, -1):          # las primeras páginas, hacia atrás
        if etiquetas[i] is None and i + 1 < n:
            etiquetas[i] = etiquetas[i + 1]
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


def cortar(pdf: bytes, desde: int, hasta: int) -> bytes:
    """Las páginas `desde`-`hasta` del PDF, como PDF aparte.

    Depurar no es sólo etiquetar: es ENTREGAR el documento limpio. El taller
    espera un PDF por cada cosa —el acto reclamado por un lado, los conceptos
    por otro— y lo que llega del Expediente Electrónico es un tomo con todo
    dentro. Aquí se corta.

    Y hay un ahorro que no es menor: la sentencia recurrida son 18 de las 117
    páginas. Mandar el tomo entero es pasar por Azure 99 páginas que no se van
    a mirar, en cada paso del taller.
    """
    import fitz
    origen = fitz.open(stream=pdf, filetype="pdf")
    try:
        n = origen.page_count
        a = max(1, min(int(desde), n))
        b = max(a, min(int(hasta), n))
        salida = fitz.open()
        try:
            salida.insert_pdf(origen, from_page=a - 1, to_page=b - 1)
            return salida.tobytes()
        finally:
            salida.close()
    finally:
        origen.close()


def repartir(segmentos):
    """Qué segmento hace de qué en el taller.

    El taller pide dos documentos: el ACTO —la sentencia que se combate— y los
    CONCEPTOS —lo que se alega contra ella—. La depuración ya sabe cuál es
    cuál; esto sólo escoge, de entre los candidatos, el más largo, porque el
    de verdad siempre lo es: los otros son portadas que lo mencionan.

    Devuelve (acto, conceptos, constancias) con los segmentos, o None cuando no
    hay candidato. Que falte no se suple: se dice.
    """
    def mayor(tipos):
        cand = [s for s in segmentos if s.get("tipo") in tipos]
        return max(cand, key=lambda s: s["caracteres"]) if cand else None

    acto = mayor({"sentencia_recurrida", "sentencia", "acto_reclamado"})
    conceptos = mayor({"promocion", "agravios", "conceptos", "demanda"})

    # EN AMPARO DIRECTO NO LLEGA UNA SENTENCIA APARTE.
    #
    # La demanda TRANSCRIBE la sentencia reclamada —es como se redactan— y el
    # expediente electrónico no trae un PDF distinto con ella. Medido en el ADC
    # 536/2025: 85 páginas de demanda con la sentencia dentro, y ni un solo
    # documento clasificado como sentencia entre las ocho constancias.
    #
    # Exigir dos ficheros distintos dejaba fuera a TODOS los amparos directos.
    # Cuando no hay acto separado, el acto está dentro del escrito: se entrega
    # el mismo documento por los dos lados y se dice, porque el taller extrae
    # cosas distintas de cada uno —de uno la razón de la responsable, del otro
    # lo que se alega contra ella— y las dos están ahí.
    if acto is None and conceptos is not None:
        acto = dict(conceptos)
        acto["mismo_documento"] = True

    usados = {id(x) for x in (acto, conceptos) if x}
    resto = [s for s in segmentos if id(s) not in usados
             and s["caracteres"] > 800]
    return acto, conceptos, resto
