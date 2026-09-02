#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paso 1 de 3 — el censo: qué tesis hay que traer.

Recorre `jurisprudencia_nacional_v3` en Qdrant y saca la lista de registros
únicos. Es el universo de lo que el chat puede llegar a citar, y por tanto de
lo que tiene que estar en nuestro repositorio para que el visor no dependa del
Semanario.

Sale un JSON con el registro y los datos que hacen falta para el cableado
posterior (rubro, clave de tesis, instancia), no sólo el número: cuando el
PDF esté en el bucket habrá que poder decir de quién es sin volver a Qdrant.

    python3 tesis_pdf_1_registros.py
"""
import io, os, json, time, urllib.request

REPO = os.path.dirname(os.path.abspath(__file__))
DESTINO = '/Volumes/KINGSTON/iurexia-tesis'
COLECCION = 'jurisprudencia_nacional_v3'


def env(nombre):
    for linea in io.open(os.path.join(REPO, '.env'), encoding='utf-8'):
        if linea.startswith(nombre + '='):
            return linea.split('=', 1)[1].strip().strip('"').strip("'")
    raise SystemExit('falta ' + nombre)


URL = env('QDRANT_URL').rstrip('/')
CLAVE = env('QDRANT_API_KEY')


def pedir(ruta, cuerpo=None):
    req = urllib.request.Request(
        URL + ruta,
        data=json.dumps(cuerpo).encode('utf-8') if cuerpo is not None else None,
        headers={'api-key': CLAVE, 'Content-Type': 'application/json'},
        method='POST' if cuerpo is not None else 'GET',
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode('utf-8'))


def main():
    os.makedirs(DESTINO, exist_ok=True)
    vistos = {}
    siguiente = None
    paginas = 0
    t0 = time.time()

    while True:
        cuerpo = {
            'limit': 1000,
            'with_payload': ['registro', 'rubro', 'clave_tesis', 'instancia',
                             'epoca', 'tipo', 'materia', 'localizacion'],
            'with_vector': False,
        }
        if siguiente is not None:
            cuerpo['offset'] = siguiente

        r = pedir('/collections/%s/points/scroll' % COLECCION, cuerpo)['result']
        for p in r['points']:
            pay = p.get('payload') or {}
            reg = str(pay.get('registro') or '').strip()
            # Sin registro no hay PDF que pedir: el Semanario indexa por él.
            if not reg.isdigit() or not (5 <= len(reg) <= 8):
                continue
            if reg in vistos:
                continue
            vistos[reg] = {
                'registro': reg,
                'rubro': (pay.get('rubro') or '')[:400],
                'clave_tesis': pay.get('clave_tesis'),
                'instancia': pay.get('instancia'),
                'epoca': pay.get('epoca'),
                'tipo': pay.get('tipo'),
                'materia': pay.get('materia'),
                'localizacion': pay.get('localizacion'),
            }

        paginas += 1
        siguiente = r.get('next_page_offset')
        print('  página %3d · %6d registros únicos · %4.0f s'
              % (paginas, len(vistos), time.time() - t0), flush=True)
        if siguiente is None:
            break

    salida = os.path.join(DESTINO, 'registros.json')
    with io.open(salida, 'w', encoding='utf-8') as f:
        json.dump(sorted(vistos.values(), key=lambda x: int(x['registro'])),
                  f, ensure_ascii=False)

    print('\n%d registros únicos de %s' % (len(vistos), COLECCION))
    print('→ %s' % salida)
    print('Estimado en disco a ~570 KB por PDF: %.1f GB' % (len(vistos) * 570_000 / 1e9))


if __name__ == '__main__':
    main()
