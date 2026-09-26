#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paso 6 — esperar a que termine la recolección y rematar solo.

Vigila cuántos PDF hay en disco. Cuando el número deja de crecer durante un
rato —señal de que el navegador acabó— lanza las dos fases que faltan: subir al
bucket y cablear `pdf_url` en Qdrant.

POR QUÉ POR CONTEO Y NO PREGUNTÁNDOLE AL NAVEGADOR: el recolector vive en una
pestaña, y una pestaña se cierra, se duerme o se recarga. El disco no miente y
no depende de nadie. Si la recolección se corta a medias, esto sube lo que haya
—que es lo correcto— en vez de quedarse esperando para siempre.

El cableado va con calma a propósito: la tanda de 63,172 de esta madrugada
disparó la alerta de CPU del clúster de Qdrant. Ocho mil son un octavo de
aquello, pero no hay prisa que lo justifique.

    ./.venv/bin/python3 tesis_pdf_6_rematar.py
"""
import os, subprocess, sys, time

REPO = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.environ.get('IUREXIA_TESIS_DIR',
                      os.path.expanduser('~/Documents/KINGSTON/iurexia-tesis'))
PDFS = os.path.join(RAIZ, 'pdf')
# El paso 3 necesita `google-cloud-storage`, que está instalado en el Python
# del sistema (--user) y NO en el entorno del repositorio. Usar el del repo
# fallaría con ModuleNotFoundError justo al final de dos horas de descarga.
PYTHON = 'python3'

QUIETO_PARA_TERMINAR = 6 * 60      # segundos sin un solo archivo nuevo
LATIDO = 30


def en_disco():
    n = 0
    for _, _, archivos in os.walk(PDFS):
        n += sum(1 for a in archivos if a.endswith('.pdf') and not a.startswith('._'))
    return n


def correr(titulo, args):
    print(f'\n══ {titulo} ══', flush=True)
    r = subprocess.run([PYTHON, os.path.join(REPO, args[0])] + args[1:],
                       cwd=REPO, capture_output=True, text=True,
                       env={**os.environ, 'PYTHONWARNINGS': 'ignore'})
    salida = (r.stdout or '') + (r.stderr or '')
    print('\n'.join(l for l in salida.splitlines() if 'Warning' not in l)[-2500:], flush=True)
    return r.returncode == 0


def main():
    print(f'Esperando a que la recolección termine · {en_disco():,} en disco', flush=True)
    ultimo, quieto_desde = en_disco(), time.time()

    while True:
        time.sleep(LATIDO)
        ahora = en_disco()
        if ahora != ultimo:
            faltan = 71655 - ahora
            print(f'  {ahora:,}/71,655 · faltan {faltan:,}', flush=True)
            ultimo, quieto_desde = ahora, time.time()
            continue
        quieto = time.time() - quieto_desde
        if quieto >= QUIETO_PARA_TERMINAR:
            print(f'\n{ahora:,} en disco y {quieto/60:.0f} min sin novedad: '
                  f'se da por terminada la recolección.', flush=True)
            break

    correr('FASE 1 · subir al bucket',
           ['tesis_pdf_3_subir_y_cablear.py', '--subir', '--hilos', '48'])
    # Con calma: la tanda anterior disparó la alerta de CPU de Qdrant.
    correr('FASE 2 · cablear con la cita',
           ['tesis_pdf_3_subir_y_cablear.py', '--cablear'])
    print(f'\nTERMINADO · {en_disco():,} PDF en disco', flush=True)


if __name__ == '__main__':
    main()
