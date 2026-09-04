#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lector de expedientes del Poder Judicial de la Federación.

QUÉ LEE Y POR DÓNDE. La página pública de captura del propio Consejo de la
Judicatura, la misma que abre cualquiera que consulte su asunto:

    https://www.dgej.cjf.gob.mx/siseinternet/Reportes/VerCaptura.aspx
        ?tipoasunto=1&organismo=293&expediente=71/2026&tipoprocedimiento=0

Devuelve la carátula —órgano, NEUN, número asignado— y la tabla `grvAcuerdos`
con la historia cronológica completa: una petición trae todo el expediente, no
sólo lo de hoy. Eso es lo que permite que un día inhábil o una caída del portal
no pierdan nada: al día siguiente vuelve a aparecer lo publicado en el hueco,
con su fecha de auto real.

LO QUE NO HACE, Y NO VA A HACER. No resuelve CAPTCHAs. Las pantallas de
búsqueda del PJF («¿qué publicó hoy este juzgado?») llevan reCAPTCHA y no se
usan: no hacen falta, porque la pregunta de un litigante es por su propio
número de expediente y para ésa la ruta de arriba está abierta. Si algún día
le ponen reto a esta también, el sistema NO intenta pasarlo: pasa a modo
asistido y avisa.

LA REGLA QUE GOBIERNA ESTE MÓDULO. Ausencia de filas NO es «sin novedad». Si la
carátula no es del expediente que pedimos, si la tabla no está, o si trae menos
acuerdos de los que ya teníamos guardados, esto NO devuelve una lectura vacía:
levanta `ErrorFormato`. Un falso «sin novedad» es indistinguible del silencio
bueno, y el silencio bueno es lo que le promete el correo al abogado.

    ./seg_lector_pjf.py 293 71/2026            # una consulta, a mano
    ./seg_lector_pjf.py 293 71/2026 --json
"""
import hashlib
import html as _html
import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request

BASE = 'https://www.dgej.cjf.gob.mx/siseinternet/Reportes/VerCaptura.aspx'
VER_ACUERDO = 'https://www.dgej.cjf.gob.mx/siseinternet/Actuaria/VerAcuerdo.aspx'
AGENTE = 'Iurexia/1.0 (+https://iurexia.com/bot; contacto: soporte@iurexia.com)'

# Las seis columnas que la tabla tiene que traer, en este orden. Si el portal
# cambia el encabezado, el parser se entera aquí y no más adelante leyendo
# fechas de la columna equivocada.
COLUMNAS = ['no.', 'fecha del auto', 'tipo cuaderno',
            'fecha de publicacion', 'resumen', 'ver sintesis completa']


class ErrorFormato(Exception):
    """La página respondió, pero no es lo que esperábamos leer."""


class ErrorNoEncontrado(Exception):
    """La página respondió y dice que ese expediente no está ahí."""


# ── Normalización y huellas ───────────────────────────────────────────

def sin_tildes(t):
    return ''.join(c for c in unicodedata.normalize('NFD', t)
                   if unicodedata.category(c) != 'Mn')


def normalizar(t):
    """Texto comparable. Se conservan tildes y eñes a propósito: en un acuerdo
       «anos» y «años» no son lo mismo, y confundirlos sería peor que el ruido
       que ahorraría."""
    t = unicodedata.normalize('NFKC', t or '')
    t = t.replace(' ', ' ')
    t = re.sub(r'[“”«»]', '"', t)
    t = re.sub(r'[‘’]', "'", t)
    # Los sellos de hora se reimprimen distintos entre visitas y no dicen nada.
    t = re.sub(r'\b\d{1,2}:\d{2}(:\d{2})?\s*(hrs?|horas)?\b', ' ', t, flags=re.I)
    return re.sub(r'\s+', ' ', t).strip().lower()


def _sha(*partes):
    return hashlib.sha256('|'.join(partes).encode('utf-8')).hexdigest()


def huella_clave(jurisdiccion, organo, numero, neun, cuaderno, fecha_auto, resumen):
    """La identidad del acuerdo.

    Lleva `fecha_auto` y no la de publicación porque la del auto es del juez y
    no se mueve. Lleva `cuaderno` porque el principal y el incidente de
    suspensión pueden tener autos el mismo día y sin esto se pisarían. Y NO
    lleva el número de orden de la lista: el juzgado lo renumera al intercalar
    un acuerdo atrasado, y si formara parte de la identidad, un día cualquiera
    el expediente entero parecería nuevo.
    """
    return _sha(jurisdiccion, str(organo), numero, neun or '',
                normalizar(cuaderno), fecha_auto or '', normalizar(resumen)[:300])


def huella_texto(resumen, texto_completo=None):
    return _sha(normalizar(resumen), normalizar(texto_completo or ''))


def simhash64(texto):
    """Simhash de 64 bits sobre trigramas de palabra.

    Sirve para el desempate: si la huella de identidad no existe pero hay un
    acuerdo del mismo cuaderno y la misma fecha muy parecido, es una reedición
    con la cabecera tocada, no un acuerdo nuevo. Sin esto, corregir un nombre
    en el resumen dispararía un correo falso.
    """
    palabras = normalizar(texto).split()
    if len(palabras) < 3:
        palabras = palabras or ['']
        trigramas = [' '.join(palabras)]
    else:
        trigramas = [' '.join(palabras[i:i + 3])
                     for i in range(len(palabras) - 2)]
    pesos = [0] * 64
    for t in trigramas:
        h = int.from_bytes(hashlib.blake2b(t.encode(), digest_size=8).digest(), 'big')
        for b in range(64):
            pesos[b] += 1 if (h >> b) & 1 else -1
    v = 0
    for b in range(64):
        if pesos[b] > 0:
            v |= 1 << b
    # Postgres bigint es con signo: se pasa al rango negativo si hace falta.
    return v - (1 << 64) if v >= (1 << 63) else v


def distancia(a, b):
    return bin((a ^ b) & ((1 << 64) - 1)).count('1')


# ── Petición ──────────────────────────────────────────────────────────

def url_de(organismo, expediente, tipo_asunto=1, tipo_procedimiento=0):
    from urllib.parse import quote
    return (f'{BASE}?tipoasunto={tipo_asunto}&organismo={organismo}'
            f'&expediente={quote(str(expediente), safe="/")}'
            f'&tipoprocedimiento={tipo_procedimiento}')


def pedir(organismo, expediente, tipo_asunto=1, tipo_procedimiento=0, timeout=45):
    """Devuelve (html, http_status, bytes). Levanta urllib.error en fallo de red."""
    url = url_de(organismo, expediente, tipo_asunto, tipo_procedimiento)
    req = urllib.request.Request(url, headers={
        'User-Agent': AGENTE,
        'Accept': 'text/html,application/xhtml+xml',
        'Accept-Language': 'es-MX,es;q=0.9',
    })
    with urllib.request.urlopen(req, timeout=timeout) as r:
        crudo = r.read()
        return crudo.decode('utf-8', 'replace'), r.status, len(crudo)


# ── Parser ────────────────────────────────────────────────────────────

def _texto(fragmento):
    return re.sub(r'\s+', ' ',
                  _html.unescape(re.sub(r'<[^>]+>', ' ', fragmento))).strip()


def _span(s, ident):
    m = re.search(rf'<span[^>]*id="{ident}"[^>]*>(.*?)</span>', s, re.S | re.I)
    return _texto(m.group(1)) if m else None


def _fecha(t):
    """«19-01-2026» → «2026-01-19». Devuelve None si no es una fecha."""
    m = re.match(r'^\s*(\d{2})-(\d{2})-(\d{4})\s*$', t or '')
    return f'{m.group(3)}-{m.group(2)}-{m.group(1)}' if m else None


def _mismo_numero(a, b):
    """«71/2026» == «071/2026». El portal a veces rellena con ceros."""
    def limpia(x):
        m = re.match(r'^\s*0*(\d+)\s*/\s*(\d{4})\s*$', x or '')
        return f'{m.group(1)}/{m.group(2)}' if m else re.sub(r'\s+', '', (x or '').lower())
    return limpia(a) == limpia(b)


def parsear(s, expediente_pedido, organismo=None):
    """Lee la página y devuelve {'caratula': {...}, 'acuerdos': [...]}.

    Levanta ErrorNoEncontrado o ErrorFormato antes que devolver algo dudoso.
    """
    if re.search(r'recaptcha|g-recaptcha|captcha', s, re.I):
        raise ErrorFormato('la página trae un reto CAPTCHA que antes no tenía')

    # El portal es explícito cuando el expediente no está: emite un alert de
    # JavaScript diciéndolo. Es la señal más limpia que da, mejor que inferirlo
    # de una carátula vacía, y su texto es justo el que hay que enseñarle al
    # abogado en el alta. Esto NO es un fallo del portal: es un dato mal dado.
    m = re.search(r'alert\(\s*["\'](No existe[^"\']{0,240})["\']\s*\)', s, re.I)
    if m:
        raise ErrorNoEncontrado(_html.unescape(m.group(1)).strip())

    organo = _span(s, 'lblNombreOrgano')
    neun = _span(s, 'lblNEUN')
    asignado = _span(s, 'lblNoExpedienteAsignado')

    if not organo and not asignado:
        # Ni carátula ni tabla: o el expediente no existe en ese órgano, o el
        # tipo de asunto es otro. Es cosa del alta, no un fallo del portal.
        if 'grvAcuerdos' not in s:
            raise ErrorNoEncontrado(
                'la página no trae carátula ni tabla de acuerdos')
        raise ErrorFormato('falta la carátula pero hay tabla')

    # La comprobación que impide leer el expediente del vecino.
    if asignado and not _mismo_numero(asignado, expediente_pedido):
        raise ErrorNoEncontrado(
            f'se pidió {expediente_pedido} y la página devolvió {asignado}')

    i = s.find('id="grvAcuerdos"')
    if i < 0:
        raise ErrorFormato('no está la tabla grvAcuerdos')
    j = s.find('</table>', i)
    tabla = s[i:j if j > 0 else len(s)]

    filas = re.findall(r'(?s)<tr[^>]*>(.*?)</tr>', tabla)
    if not filas:
        raise ErrorFormato('la tabla grvAcuerdos no tiene filas')

    encabezado = [sin_tildes(_texto(c)).lower()
                  for c in re.findall(r'(?s)<t[dh][^>]*>(.*?)</t[dh]>', filas[0])]
    if encabezado != COLUMNAS:
        raise ErrorFormato(
            f'el encabezado cambió: esperaba {COLUMNAS}, llegó {encabezado}')

    acuerdos = []
    for fila in filas[1:]:
        celdas = re.findall(r'(?s)<t[dh][^>]*>(.*?)</t[dh]>', fila)
        if len(celdas) != 6:
            continue                      # filas de paginación del GridView
        crudas = [_texto(c) for c in celdas]
        if not crudas[4]:
            continue                      # sin resumen no hay acuerdo que contar

        # Los parámetros para reconstruir el enlace al texto íntegro salen del
        # propio javascript del enlace «Ver síntesis».
        m = re.search(r'DoVerAcuerdo\(([^)]*)\)', celdas[5])
        args = []
        if m:
            args = [a.strip().strip('"').strip("'")
                    for a in _html.unescape(m.group(1)).split(',')]

        acuerdos.append({
            'orden_en_lista': int(crudas[0]) if crudas[0].isdigit() else None,
            'fecha_auto': _fecha(crudas[1]),
            'cuaderno': crudas[2] or None,
            'fecha_publicacion': _fecha(crudas[3]),
            'resumen': crudas[4],
            'args_veracuerdo': args or None,
        })

    if not acuerdos:
        # Un expediente recién ingresado puede no tener acuerdos todavía, pero
        # entonces la carátula sí está. Se devuelve vacío y que decida quien
        # compara: si antes había acuerdos y ahora no, eso lo detecta el
        # llamador y es fallo, no «sin novedad».
        pass

    return {
        'caratula': {
            'organo': organo,
            'neun': neun,
            'expediente': asignado,
            'control_occ': _span(s, 'lblNoControlOCC'),
            'organismo': str(organismo) if organismo is not None else None,
        },
        'acuerdos': acuerdos,
    }


def con_huellas(lectura, jurisdiccion='PJF'):
    """Añade a cada acuerdo sus tres huellas. Es lo que consume el detector."""
    c = lectura['caratula']
    for a in lectura['acuerdos']:
        a['huella_clave'] = huella_clave(
            jurisdiccion, c.get('organismo') or '', c.get('expediente') or '',
            c.get('neun'), a.get('cuaderno') or '', a.get('fecha_auto'),
            a.get('resumen') or '')
        a['huella_texto'] = huella_texto(a.get('resumen') or '')
        a['simhash'] = simhash64(a.get('resumen') or '')
    return lectura


def leer(organismo, expediente, tipo_asunto=1, tipo_procedimiento=0,
         minimo_esperado=0):
    """La operación completa: pedir, parsear, huellar y comprobar la forma.

    `minimo_esperado` son los acuerdos que ya teníamos guardados. Si la página
    trae menos, algo va mal en el portal o en el parser, y eso NO se puede
    reportar como «sin novedad».
    """
    s, status, bytes_ = pedir(organismo, expediente, tipo_asunto, tipo_procedimiento)
    lectura = con_huellas(parsear(s, expediente, organismo))

    n = len(lectura['acuerdos'])
    if minimo_esperado and n < minimo_esperado:
        raise ErrorFormato(
            f'la página trajo {n} acuerdos y ya teníamos {minimo_esperado}: '
            'no se puede afirmar que no hubo novedad')

    lectura['http_status'] = status
    lectura['bytes'] = bytes_
    lectura['hash_respuesta'] = hashlib.sha256(s.encode('utf-8')).hexdigest()
    lectura['url'] = url_de(organismo, expediente, tipo_asunto, tipo_procedimiento)
    return lectura


# ── A mano ────────────────────────────────────────────────────────────

def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    if len(args) < 2:
        print(__doc__.strip().splitlines()[-3].strip())
        sys.exit(2)
    organismo, expediente = args[0], args[1]
    tipo = int(args[2]) if len(args) > 2 else 1

    t0 = time.time()
    try:
        l = leer(organismo, expediente, tipo)
    except (ErrorFormato, ErrorNoEncontrado) as e:
        print(f'✗ {type(e).__name__}: {e}')
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f'✗ red: {e}')
        sys.exit(1)

    if '--json' in sys.argv:
        print(json.dumps(l, ensure_ascii=False, indent=1))
        return

    c = l['caratula']
    print(f"{c['organo']}")
    print(f"  expediente {c['expediente']} · NEUN {c['neun']} · "
          f"{len(l['acuerdos'])} acuerdos · {l['bytes']:,} bytes · "
          f"{time.time()-t0:.1f} s\n")
    for a in l['acuerdos'][-6:]:
        print(f"  {a['fecha_auto']}  {(a['cuaderno'] or '')[:22]:24} "
              f"{a['resumen'][:74]}")
        print(f"  {'':12}  clave {a['huella_clave'][:12]}…  simhash {a['simhash']}")


if __name__ == '__main__':
    main()
