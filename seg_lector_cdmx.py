#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lector del Boletín Judicial de la Ciudad de México.

POR QUÉ POR BOLETÍN Y NO POR BUSCADOR. El buscador de expedientes del TSJCDMX
(SICOR) tiene doble muro: un captcha dibujado en canvas por su propio
JavaScript y además reCAPTCHA v3. No se usa. No hace falta: el Boletín Judicial
publica CADA DÍA, en un solo PDF, todos los acuerdos de todos los juzgados, y
eso es un documento público por disposición legal. Una descarga al día resuelve
la cartera entera, sin volver a tocar la red por cada expediente.

CÓMO ES EL PDF. Unos 20 MB, 400 páginas, %PDF-1.7 cifrado con contraseña de
PROPIETARIO —se abre sin contraseña de usuario— y CON capa de texto, así que
no hace falta OCR. Publicado sobre las 05:49 hora de la Ciudad de México, tres
horas y veinte minutos antes de nuestra revisión de las 9:10.

CÓMO SE ESTRUCTURA POR DENTRO. Por juzgado, y dentro de cada uno por
secretaría:

    CUARTO DE LO CIVIL
    SECRETARÍA "A"
    ACUERDOS DEL 1 DE SEPTIEMBRE DEL 2026
    Fulano vs. Mengano. Ord. Civil Acuerdo. 1 Acdo. Núm. Exp. 1053/2024.
    ...

El número de expediente cierra cada entrada tras «Núm. Exp.». El juzgado va en
ordinal escrito —«CUARTO DE LO CIVIL»— y es imprescindible: el expediente
200/2026 existe en muchos juzgados a la vez, y sin el juzgado el seguimiento
avisaría del asunto de otro.

    ./seg_lector_cdmx.py --indice
    ./seg_lector_cdmx.py --buscar "cuarto de lo civil" 1053/2024
"""
import io
import os
import re
import sys
import unicodedata
import urllib.request

INDICE = 'https://consultabpj.poderjudicialcdmx.gob.mx:2096/consultaboletinpjcdmx'
AGENTE = 'Iurexia/1.0 (+https://iurexia.com/bot; contacto: soporte@iurexia.com)'

MESES = {'ene': 1, 'feb': 2, 'mar': 3, 'abr': 4, 'may': 5, 'jun': 6,
         'jul': 7, 'ago': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dic': 12}

ORDINALES = ('PRIMERO|SEGUNDO|TERCERO|CUARTO|QUINTO|SEXTO|SÉPTIMO|SEPTIMO|OCTAVO|'
             'NOVENO|DÉCIMO|DECIMO|UNDÉCIMO|UNDECIMO|DUODÉCIMO|DUODECIMO|'
             'VIGÉSIMO|VIGESIMO|TRIGÉSIMO|TRIGESIMO|CUADRAGÉSIMO|CUADRAGESIMO|'
             'QUINCUAGÉSIMO|QUINCUAGESIMO|SEXAGÉSIMO|SEXAGESIMO|'
             'SEPTUAGÉSIMO|SEPTUAGESIMO')
# La cabecera del juzgado: un ordinal (a veces compuesto, «VIGÉSIMO PRIMERO»)
# seguido de la materia. Va en línea propia y en mayúsculas.
RE_JUZGADO = re.compile(
    rf'^\s*((?:{ORDINALES})(?:\s+(?:{ORDINALES}))?\s+DE\s+LO\s+'
    r'[A-ZÁÉÍÓÚÑ ]{4,40})\s*$')
# El PDF parte el rótulo del juzgado en varias líneas cuando no le cabe:
# «CUADRAGÉSIMO PRIMERO DE LO CIVIL DE» / «PROCESO ORAL». Sin unirlas, el
# mismo juzgado queda indexado bajo tres claves distintas y una búsqueda por
# su nombre completo no lo encuentra.
RE_CONTINUA = re.compile(r'^\s*([A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ ]{2,40})\s*$')
RE_SECRETARIA = re.compile(r'^\s*SECRETAR[IÍ]A\s*[“"«]?\s*([A-Z])\s*[”"»]?\s*$')
RE_ACUERDOS_DEL = re.compile(r'^\s*ACUERDOS?\s+DEL?\s+(.{6,60})\s*$', re.I)
# El expediente cierra la entrada. Tolera «Tomo III» detrás.
RE_EXP = re.compile(r'Núm\.\s*Exp\.\s*(\d{1,6})\s*/\s*(\d{4})', re.I)


class ErrorBoletin(Exception):
    pass


def llano(t):
    t = ''.join(c for c in unicodedata.normalize('NFD', t or '')
                if unicodedata.category(c) != 'Mn')
    return re.sub(r'[^a-z0-9]+', ' ', t.lower()).strip()


def _pedir(url, timeout=300):
    req = urllib.request.Request(url, headers={'User-Agent': AGENTE})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def indice():
    """Los boletines publicados, del más reciente al más antiguo.

    Devuelve [{'fecha': 'aaaa-mm-dd', 'id_externo': '4092', 'url': '…pdf'}].
    No hace falta POST ni token CSRF: la página trae fecha y PDF en cada fila.
    """
    s = _pedir(INDICE, timeout=60).decode('utf-8', 'replace')
    salida = []
    for fila in re.findall(r'(?s)<tr[^>]*>(.*?)</tr>', s):
        f = re.search(r'(\d{2})-([a-zA-Z]{3})\.?-(\d{4})', fila)
        pdf = re.search(r'href="([^"]*/pdf/boletines/[^"]+\.pdf)"', fila)
        ext = re.search(r'/externo/(\d+)', fila)
        if not (f and pdf):
            continue
        mes = MESES.get(f.group(2)[:3].lower())
        if not mes:
            continue
        salida.append({'fecha': f'{f.group(3)}-{mes:02d}-{int(f.group(1)):02d}',
                       'id_externo': ext.group(1) if ext else None,
                       'url': pdf.group(1)})
    if not salida:
        raise ErrorBoletin('el índice no trae ninguna fila con fecha y PDF')
    return salida


def del_dia(fecha):
    """El boletín de una fecha concreta, o None si ese día no se publicó."""
    return next((b for b in indice() if b['fecha'] == fecha), None)


def descargar(url, destino):
    datos = _pedir(url)
    if not datos.startswith(b'%PDF'):
        raise ErrorBoletin(f'lo descargado no es un PDF ({len(datos)} bytes)')
    with open(destino, 'wb') as f:
        f.write(datos)
    return len(datos)


def indexar(ruta_pdf):
    """Lee el PDF y devuelve {(juzgado_llano, 'N/AAAA'): [entradas]}.

    Cada entrada trae el juzgado tal cual, la secretaría, la fecha de los
    acuerdos, la página y el texto íntegro de la anotación.
    """
    import pypdf
    lector = pypdf.PdfReader(ruta_pdf)
    if lector.is_encrypted and lector.decrypt('') == 0:
        raise ErrorBoletin('el PDF pide contraseña de usuario')

    if len(lector.pages) < 20:
        raise ErrorBoletin(f'sólo {len(lector.pages)} páginas: no parece el boletín')

    indice_exp, juzgado, secretaria, acuerdos_del = {}, None, None, None
    vistos_juzgados = set()
    esperando_resto = False

    for n, pagina in enumerate(lector.pages, 1):
        texto = pagina.extract_text() or ''
        # Se acumula por párrafos: una entrada puede ocupar varias líneas y
        # cierra en «Núm. Exp. N/AAAA».
        buffer = []
        for linea in texto.splitlines():
            l = linea.strip()
            if not l:
                continue

            m = RE_JUZGADO.match(l)
            if m:
                juzgado = re.sub(r'\s+', ' ', m.group(1)).strip()
                esperando_resto = True
                buffer = []
                continue
            if esperando_resto:
                # Sólo continúa si la línea es rótulo en mayúsculas; en cuanto
                # llega texto normal, el nombre del juzgado está completo.
                if RE_CONTINUA.match(l) and not RE_SECRETARIA.match(l) \
                        and not RE_ACUERDOS_DEL.match(l):
                    juzgado = f'{juzgado} {l.strip()}'.strip()
                    continue
                esperando_resto = False
                if juzgado:
                    vistos_juzgados.add(juzgado)
            m = RE_SECRETARIA.match(l)
            if m:
                secretaria = m.group(1)
                buffer = []
                continue
            m = RE_ACUERDOS_DEL.match(l)
            if m:
                acuerdos_del = re.sub(r'\s+', ' ', m.group(1)).strip(' .')
                buffer = []
                continue

            buffer.append(l)
            m = RE_EXP.search(l)
            if m and juzgado:
                entrada = re.sub(r'\s+', ' ', ' '.join(buffer)).strip()
                # El «Tomo II.» que cierra la entrada anterior cae al principio
                # de ésta, porque va detrás del número en la misma línea.
                entrada = re.sub(r'^(Tomo\s+[IVXLC]+\.?\s*)+', '', entrada).strip()
                clave = (llano(juzgado), f'{int(m.group(1))}/{m.group(2)}')
                indice_exp.setdefault(clave, []).append({
                    'juzgado': juzgado, 'secretaria': secretaria,
                    'acuerdos_del': acuerdos_del, 'pagina': n,
                    'texto': entrada,
                })
                buffer = []

    if not indice_exp:
        raise ErrorBoletin('no se reconoció ninguna entrada: ¿cambió el formato?')

    return {'entradas': indice_exp, 'juzgados': sorted(vistos_juzgados),
            'paginas': len(lector.pages)}


def buscar(indexado, juzgado, expediente):
    """Las entradas de ese expediente en ese juzgado. Lista vacía si no salió."""
    n, a = expediente.split('/')
    return indexado['entradas'].get((llano(juzgado), f'{int(n)}/{a}'), [])


def main():
    if '--indice' in sys.argv:
        for b in indice()[:10]:
            print(f"  {b['fecha']}  id {b['id_externo'] or '—':>5}  {b['url']}")
        return

    if '--buscar' in sys.argv:
        i = sys.argv.index('--buscar')
        juzgado, expediente = sys.argv[i + 1], sys.argv[i + 2]
        b = indice()[0]
        ruta = f'/tmp/boletin_{b["fecha"]}.pdf'
        if not os.path.exists(ruta):
            print(f'descargando el boletín del {b["fecha"]}…')
            descargar(b['url'], ruta)
        idx = indexar(ruta)
        print(f"{b['fecha']} · {idx['paginas']} páginas · "
              f"{len(idx['juzgados'])} juzgados · "
              f"{sum(len(v) for v in idx['entradas'].values()):,} entradas\n")
        hits = buscar(idx, juzgado, expediente)
        if not hits:
            print(f'  sin entradas para {expediente} en «{juzgado}»')
            return
        for h in hits:
            print(f"  pág {h['pagina']} · {h['juzgado']} · Secretaría {h['secretaria']}")
            print(f"  acuerdos del {h['acuerdos_del']}")
            print(f"  {h['texto'][:260]}\n")
        return

    print(__doc__.strip())


if __name__ == '__main__':
    main()
