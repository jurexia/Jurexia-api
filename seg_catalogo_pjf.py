#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cosecha el catálogo de órganos jurisdiccionales del PJF para `seg_organos`.

DE DÓNDE SALE. De la misma página pública que usa cualquiera para consultar un
expediente:

    https://www.dgej.cjf.gob.mx/internet/expedientes/circuitos.asp?Cir={id}&Exp=1

Devuelve, para un circuito, el nombre del circuito en un campo oculto y la
lista de sus órganos como <option value="{CatOrganismoId}">{nombre}</option>.
Ese `value` es exactamente el `organismo=` que después pide `VerCaptura.aspx`,
así que cosechar aquí es lo que permite consultar allí.

POR QUÉ SE RECORREN IDS A CIEGAS. Porque el id interno del circuito NO es el
ordinal romano: el Vigésimo Segundo (Querétaro) es el 53. La numeración tiene
huecos, así que se prueba un rango y se conserva lo que responde con órganos.
Confundir id con ordinal es lo que rompe la consulta en la mayoría de los
circuitos, y es el error que este script existe para no cometer.

RITMO. Una petición cada 1.5 s, un solo hilo, `User-Agent` con contacto. Son
unas sesenta peticiones una vez al mes: hay un F5 delante y la manera de
convivir con él es pedir poco y decir quién eres.

    ./seg_catalogo_pjf.py            # cosecha y escribe seg_catalogo_pjf.json
    ./seg_catalogo_pjf.py --subir    # además lo carga en Supabase
"""
import html
import io
import json
import os
import re
import sys
import time
import urllib.request

BASE = 'https://www.dgej.cjf.gob.mx/internet/expedientes/circuitos.asp'
AGENTE = 'Iurexia/1.0 (+https://iurexia.com/bot; contacto: soporte@iurexia.com)'
PAUSA = 1.5
# Los ids conocidos van del 1 al 56 con huecos, más el 109. Se recorre con
# holgura por si abren circuitos nuevos; los que no existen devuelven la página
# sin <option> y se descartan solos.
CANDIDATOS = list(range(1, 61)) + [109]

AQUI = os.path.dirname(os.path.abspath(__file__))
SALIDA = os.path.join(AQUI, 'seg_catalogo_pjf.json')

ORDINALES = [
    'primer', 'segundo', 'tercer', 'cuarto', 'quinto', 'sexto', 'septimo',
    'octavo', 'noveno', 'decimo', 'decimoprimer', 'decimosegundo',
    'decimotercer', 'decimocuarto', 'decimoquinto', 'decimosexto',
    'decimoseptimo', 'decimoctavo', 'decimonoveno', 'vigesimo',
    'vigesimo primer', 'vigesimo segundo', 'vigesimo tercer',
    'vigesimo cuarto', 'vigesimo quinto', 'vigesimo sexto', 'vigesimo septimo',
    'vigesimo octavo', 'vigesimo noveno', 'trigesimo', 'trigesimo primer',
    'trigesimo segundo',
]


def sin_tildes(t):
    import unicodedata
    return ''.join(c for c in unicodedata.normalize('NFD', t)
                   if unicodedata.category(c) != 'Mn')


def ordinal_de(nombre_circuito):
    """«VIGÉSIMO SEGUNDO CIRCUITO» → 22. Devuelve None si no se reconoce."""
    # El sitio mezcla las dos grafías: escribe «DECIMOPRIMER CIRCUITO» en unos
    # y «DÉCIMO CUARTO CIRCUITO» en otros. Se comparan sin espacios para que
    # «decimo cuarto» y «decimocuarto» sean la misma cosa.
    def llana(t):
        t = re.sub(r'[^a-z]', '', sin_tildes(t).lower().replace(' circuito', ''))
        # «décimo octavo» junto da «decimooctavo»; la grafía correcta es
        # «decimoctavo». Ningún otro ordinal lleva doble o, así que colapsarla
        # es seguro y cierra el único caso que quedaba fuera.
        return t.replace('oo', 'o')

    n = llana(nombre_circuito)
    for i, o in enumerate(ORDINALES, 1):
        if n == llana(o):
            return i
    return None


def familia_de(nombre):
    """Qué clase de órgano es. Manda qué tipos de asunto se le pueden pedir."""
    n = sin_tildes(nombre).lower()
    if 'tribunal colegiado de apelacion' in n:
        return 'tca'
    if 'tribunal colegiado' in n:
        return 'tcc'
    if 'tribunal laboral' in n:
        return 'tribunal_laboral'
    if 'pleno regional' in n or 'plenos de circuito' in n:
        return 'pleno_regional'
    if 'centro de justicia penal' in n:
        return 'cjpf'
    if 'juzgado' in n and 'distrito' in n:
        return 'juzgado_distrito'
    return 'otro'


def materia_de(nombre):
    n = sin_tildes(nombre).lower()
    # El orden importa: «amparo civil, administrativo y de trabajo» es mixta,
    # y etiquetarla como Civil a secas engañaría en el buscador del alta.
    if 'amparo civil, administrativo y de trabajo' in n:
        return 'Mixta (amparo)'
    for clave, etiqueta in (
        ('materia penal', 'Penal'), ('materia civil', 'Civil'),
        ('materia administrativa', 'Administrativa'),
        ('materia de trabajo', 'Trabajo'), ('materia laboral', 'Trabajo'),
        ('materia mercantil', 'Mercantil'), ('materia mixta', 'Mixta'),
    ):
        if clave in n:
            return etiqueta
    return None


def vigencia_de(nombre):
    """El PJF rotula los órganos extintos dentro del propio nombre:
       «Juzgado … (01/01/2010 - 30/09/2024)». Se extrae la fecha final."""
    m = re.search(r'-\s*(\d{2})/(\d{2})/(\d{4})\s*\)', nombre)
    return f'{m.group(3)}-{m.group(2)}-{m.group(1)}' if m else None


def pedir(url):
    req = urllib.request.Request(url, headers={'User-Agent': AGENTE})
    with urllib.request.urlopen(req, timeout=40) as r:
        # El sitio va en windows-1252; decodificar como utf-8 parte las eñes.
        return r.read().decode('windows-1252', 'replace')


def circuito(cir):
    """Devuelve (nombre_circuito, [(id, nombre), …]) o (None, []) si no existe."""
    s = pedir(f'{BASE}?Cir={cir}&Exp=1')
    m = re.search(r'name="CircuitoName"[^>]*value="([^"]*)"', s, re.I)
    nombre = html.unescape(m.group(1)).strip() if m else None
    organos = [(v, re.sub(r'\s+', ' ', html.unescape(t)).strip())
               for v, t in re.findall(
                   r'<option[^>]*value="(\d+)"[^>]*>([^<]+)', s, re.I)]
    return nombre, organos


def main():
    filas, mapa, fallos = [], [], []
    print(f'Cosechando {len(CANDIDATOS)} candidatos, uno cada {PAUSA} s\n')

    for cir in CANDIDATOS:
        try:
            nombre, organos = circuito(cir)
        except Exception as e:
            fallos.append((cir, f'{type(e).__name__}: {e}'))
            time.sleep(PAUSA)
            continue

        if not organos:
            time.sleep(PAUSA)
            continue

        ordinal = ordinal_de(nombre or '')
        mapa.append({'cir': cir, 'nombre': nombre, 'ordinal': ordinal,
                     'organos': len(organos)})
        print(f'  Cir={cir:<4} {(nombre or "?")[:44]:46} {len(organos):>3} órganos'
              + ('' if ordinal else '   ← ordinal no reconocido'), flush=True)

        for oid, onombre in organos:
            filas.append({
                'jurisdiccion': 'PJF',
                'clave_externa': oid,
                'nombre': onombre,
                'circuito_id': cir,
                'circuito_ordinal': ordinal,
                'familia': familia_de(onombre),
                'materia': materia_de(onombre),
                'vigencia_hasta': vigencia_de(onombre),
                'activo': vigencia_de(onombre) is None,
                'metadatos': {'circuito_nombre': nombre},
            })
        time.sleep(PAUSA)

    # Un mismo órgano puede aparecer en dos circuitos si el sitio lo repite.
    por_clave = {}
    for f in filas:
        por_clave[f['clave_externa']] = f
    filas = list(por_clave.values())

    io.open(SALIDA, 'w', encoding='utf-8').write(
        json.dumps({'mapa': mapa, 'organos': filas}, ensure_ascii=False, indent=1))

    print(f'\nCircuitos con órganos: {len(mapa)}')
    print(f'Órganos únicos       : {len(filas):,}')
    print(f'  vigentes           : {sum(1 for f in filas if f["activo"]):,}')
    print(f'  extintos           : {sum(1 for f in filas if not f["activo"]):,}')
    if fallos:
        print(f'Fallos: {len(fallos)} → {fallos[:3]}')
    print(f'Escrito en {SALIDA}')

    if '--subir' in sys.argv:
        subir(filas)


def subir(filas):
    env = {}
    for l in io.open(os.path.join(AQUI, '.env'), encoding='utf-8'):
        l = l.strip()
        if l and not l.startswith('#') and '=' in l:
            k, v = l.split('=', 1)
            env[k] = v.strip().strip('"').strip("'")
    url = env.get('SUPABASE_URL') or env['NEXT_PUBLIC_SUPABASE_URL']
    key = env['SUPABASE_SERVICE_KEY']

    guardadas = 0
    for i in range(0, len(filas), 200):
        lote = filas[i:i + 200]
        req = urllib.request.Request(
            f'{url}/rest/v1/seg_organos?on_conflict=jurisdiccion,clave_externa',
            data=json.dumps(lote, ensure_ascii=False).encode(),
            headers={'apikey': key, 'Authorization': f'Bearer {key}',
                     'Content-Type': 'application/json',
                     'Prefer': 'resolution=merge-duplicates,return=minimal'},
            method='POST')
        try:
            urllib.request.urlopen(req, timeout=90)
            guardadas += len(lote)
        except urllib.error.HTTPError as e:
            print('  fallo al subir:', e.read().decode()[:300])
            break
    print(f'Subidos a seg_organos: {guardadas:,}')


if __name__ == '__main__':
    main()
