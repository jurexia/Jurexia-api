#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paso 5 — recoger los lotes que el navegador va dejando en Descargas.

POR QUÉ POR LA CARPETA DE DESCARGAS (3-sep-2026)
-----------------------------------------------
El plan era que la página entregara cada PDF a un servidor local escuchando en
127.0.0.1. No se puede: la CSP del Semanario limita `connect-src` a
`*.scjn.gob.mx`, así que la página tiene prohibido hablar con la máquina. Ese
camino se abandonó —de ahí que no exista un paso 4—.

La salida es la propia descarga del navegador. La página junta varios PDF en un
JSON —registro → base64— y lo entrega con `<a download>`, que funciona porque
la petición es del mismo origen. Este programa vigila la carpeta de Descargas,
desempaqueta cada lote, valida y archiva.

Por qué el navegador y no `curl`: todo `scjn.gob.mx` está detrás de Incapsula y
esta IP sigue en su reto —302 en bucle— desde la descarga masiva de esta
madrugada. El navegador pasa porque resolvió el reto y guarda la cookie; es el
sitio atendiendo a un navegador, no un rodeo.

    ./.venv/bin/python3 tesis_pdf_5_desempaquetar.py           # una pasada
    ./.venv/bin/python3 tesis_pdf_5_desempaquetar.py --vigilar # hasta Ctrl-C
"""
import base64, hashlib, io, json, os, sys, time

DESCARGAS = os.path.expanduser('~/Downloads')
RAIZ = os.environ.get('IUREXIA_TESIS_DIR',
                      os.path.expanduser('~/Documents/KINGSTON/iurexia-tesis'))
PDFS = os.path.join(RAIZ, 'pdf')
MANIFIESTO = os.path.join(RAIZ, 'manifiesto.jsonl')
CONSUMIDOS = os.path.join(RAIZ, 'lotes_consumidos')
MINIMO_BYTES = 20_000
PREFIJO = 'iurexia-tesis-lote'


def ruta_de(reg):
    return os.path.join(PDFS, reg[:3], f'{reg}.pdf')


def procesar(ruta):
    """Desempaqueta un lote. Devuelve (guardados, invalidos, repetidos)."""
    try:
        with io.open(ruta, encoding='utf-8') as f:
            lote = json.load(f)
    except Exception as e:
        print(f'  ✗ {os.path.basename(ruta)}: no se pudo leer ({e})')
        return 0, 0, 0

    guardados = invalidos = repetidos = 0
    for reg, b64 in lote.items():
        if not str(reg).isdigit():
            continue
        destino = ruta_de(reg)
        try:
            if os.path.getsize(destino) >= MINIMO_BYTES:
                repetidos += 1
                continue
        except OSError:
            pass

        try:
            datos = base64.b64decode(b64)
        except Exception:
            invalidos += 1
            continue

        # La misma validación de siempre: el endpoint devuelve 200 con un PDF
        # truncado cuando el registro no existe.
        if not datos.startswith(b'%PDF') or len(datos) < MINIMO_BYTES:
            invalidos += 1
            continue

        os.makedirs(os.path.dirname(destino), exist_ok=True)
        tmp = destino + '.parcial'
        with open(tmp, 'wb') as f:
            f.write(datos)
        os.replace(tmp, destino)

        with io.open(MANIFIESTO, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'registro': reg, 'bytes': len(datos),
                                'sha1': hashlib.sha1(datos).hexdigest(),
                                'via': 'navegador'}, ensure_ascii=False) + '\n')
        guardados += 1

    # El lote se aparta, no se borra: si algo saliera mal en el archivado,
    # borrar la única copia sería perder la descarga.
    os.makedirs(CONSUMIDOS, exist_ok=True)
    os.replace(ruta, os.path.join(CONSUMIDOS, os.path.basename(ruta)))
    return guardados, invalidos, repetidos


def lotes():
    return sorted(os.path.join(DESCARGAS, f) for f in os.listdir(DESCARGAS)
                  if f.startswith(PREFIJO) and f.endswith('.json'))


def en_disco():
    n = 0
    for raiz, _, archivos in os.walk(PDFS):
        n += sum(1 for a in archivos if a.endswith('.pdf') and not a.startswith('._'))
    return n


def main():
    vigilar = '--vigilar' in sys.argv
    total_g = total_i = total_r = 0
    t0 = time.time()
    print(f'Vigilando {DESCARGAS} · archivando en {PDFS}\n', flush=True)

    while True:
        pendientes = lotes()
        for ruta in pendientes:
            # Un lote que aún se está escribiendo daría JSON truncado.
            if time.time() - os.path.getmtime(ruta) < 2:
                continue
            g, i, r = procesar(ruta)
            total_g += g; total_i += i; total_r += r
            print(f'  {os.path.basename(ruta):42} +{g:>3} · {en_disco():,}/71,655',
                  flush=True)
        if not vigilar:
            break
        time.sleep(4)

    print(f'\nGuardados: {total_g:,} · inválidos: {total_i:,} · '
          f'ya estaban: {total_r:,} · {(time.time()-t0)/60:.1f} min')
    print(f'En disco: {en_disco():,} de 71,655')


if __name__ == '__main__':
    main()
