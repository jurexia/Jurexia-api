#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paso 3 de 3 — subir al bucket y cablear el PDF con la cita.

QUÉ MECÁNICA SE SIGUE, Y POR QUÉ ÉSA
------------------------------------
La que ya funciona con las sentencias de TCC: el PDF vive en el bucket
`iurexia-leyes` y su dirección va en el payload de Qdrant, en el campo
`pdf_url`. El panel de citas del frontend lee ese campo y no necesita saber
nada más, así que las tesis quedan cableadas sin tocar una línea de la
aplicación:

    sentencias  →  iurexia-leyes/sentencias-1/8TCC_CIV/467-2024_35846084.pdf
    tesis       →  iurexia-leyes/tesis/172/172479.pdf

Se reparte por los tres primeros dígitos del registro, igual que en el disco:
71,655 objetos bajo un solo prefijo hacen incómodo hasta listarlos.

DOS FASES, Y SE PUEDEN CORRER POR SEPARADO
------------------------------------------
    --subir     lleva los PDF del Kingston al bucket
    --cablear   escribe `pdf_url` en los puntos de jurisprudencia_nacional_v3

Separarlas importa: subir es lento y reanudable, cablear es rápido y hay que
poder repetirlo si se reingesta la colección. Sin argumentos hace las dos.

Ambas son idempotentes: subir salta lo que ya está en el bucket con el mismo
tamaño, y cablear reescribe el mismo valor sin efecto.

    python3 tesis_pdf_3_subir_y_cablear.py --subir --hilos 24
    python3 tesis_pdf_3_subir_y_cablear.py --cablear
"""
import io, json, os, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

REPO = os.path.dirname(os.path.abspath(__file__))
# Dónde viven los PDF. Se movieron del USB al disco interno el 2-sep-2026:
# el Kingston se desmontó solo a mitad de la descarga —62,936 archivos
# dentro— y con él se fueron el proceso y el registro de progreso. Un
# trabajo de tres horas no puede colgar de un cable.
RAIZ = os.environ.get(
    'IUREXIA_TESIS_DIR',
    os.path.expanduser('~/Documents/KINGSTON/iurexia-tesis'))
PDFS = os.path.join(RAIZ, 'pdf')
MANIFIESTO = os.path.join(RAIZ, 'manifiesto.jsonl')

BUCKET = 'iurexia-leyes'
PREFIJO = 'tesis'
COLECCION = 'jurisprudencia_nacional_v3'
# Se sube contra GCS, no contra el Semanario: aquí no hay cortafuegos que
# aplacar, sólo latencia que tapar. Con 64 hilos la red va llena; el límite lo
# pone el ancho de subida, no el servidor.
HILOS = 64


def env(nombre):
    for linea in io.open(os.path.join(REPO, '.env'), encoding='utf-8'):
        if linea.startswith(nombre + '='):
            return linea.split('=', 1)[1].strip().strip('"').strip("'")
    raise SystemExit('falta ' + nombre)


def objeto_de(registro: str) -> str:
    return f'{PREFIJO}/{registro[:3]}/{registro}.pdf'


def url_de(registro: str) -> str:
    return f'https://storage.googleapis.com/{BUCKET}/{objeto_de(registro)}'


def leer_manifiesto():
    filas = []
    with io.open(MANIFIESTO, encoding='utf-8') as f:
        for linea in f:
            linea = linea.strip()
            if linea:
                filas.append(json.loads(linea))
    # El manifiesto se escribe en modo append y un relanzamiento puede repetir
    # una línea; gana la última, que es la del archivo que está en disco.
    unicos = {}
    for x in filas:
        unicos[x['registro']] = x
    return list(unicos.values())


# ── FASE 1: subir ────────────────────────────────────────────────────────
def subir(hilos: int):
    from google.cloud import storage
    from google.oauth2 import service_account

    credenciales = service_account.Credentials.from_service_account_info(
        json.loads(env('GCP_SA_KEY_JSON')))
    cliente = storage.Client(credentials=credenciales,
                             project=credenciales.project_id)
    bucket = cliente.bucket(BUCKET)

    filas = leer_manifiesto()

    # QUÉ HAY YA EN EL BUCKET, DE UNA SOLA VEZ.
    #
    # La primera versión preguntaba archivo por archivo con `blob.reload()`, y
    # eso es un viaje de ida y vuelta extra por cada uno de los 63,172 —el
    # doble de peticiones para averiguar algo que un listado resuelve en unas
    # decenas—. Con el prefijo recién estrenado, además, los 63,172 sondeos
    # daban «no existe»: puro peaje.
    print('leyendo lo que ya está en el bucket…', flush=True)
    t_listado = time.time()
    ya = {}
    for blob in cliente.list_blobs(BUCKET, prefix=f'{PREFIJO}/'):
        ya[blob.name] = blob.size
    print(f'  {len(ya):,} objetos ya en {BUCKET}/{PREFIJO}/ '
          f'({time.time()-t_listado:.1f} s)', flush=True)

    print(f'{len(filas):,} tesis en el manifiesto · {hilos} hilos', flush=True)

    lock = threading.Lock()
    n = {'subidos': 0, 'ya_estaban': 0, 'sin_archivo': 0, 'errores': 0, 'bytes': 0}
    t0 = time.time()

    def una(fila):
        reg = fila['registro']
        local = os.path.join(PDFS, reg[:3], f'{reg}.pdf')
        try:
            tam = os.path.getsize(local)
        except OSError:
            with lock:
                n['sin_archivo'] += 1
            return

        # Idempotencia sin coste: el listado de arriba ya dijo qué hay.
        if ya.get(objeto_de(reg)) == tam:
            with lock:
                n['ya_estaban'] += 1
            return

        blob = bucket.blob(objeto_de(reg))
        try:
            blob.cache_control = 'public, max-age=31536000, immutable'
            blob.content_disposition = f'inline; filename="tesis-{reg}.pdf"'
            blob.upload_from_filename(local, content_type='application/pdf')
            with lock:
                n['subidos'] += 1
                n['bytes'] += tam
        except Exception as e:
            with lock:
                n['errores'] += 1
            if n['errores'] <= 5:
                print(f'  error con {reg}: {type(e).__name__}: {e}', flush=True)

    parar = threading.Event()

    def reloj():
        while not parar.wait(20):
            seg = max(time.time() - t0, 1)
            hechos = n['subidos'] + n['ya_estaban'] + n['sin_archivo'] + n['errores']
            print(f'  {n["subidos"]:>6,} subidos · {n["ya_estaban"]:>6,} ya estaban · '
                  f'{n["bytes"]/1e9:>5.2f} GB · {n["subidos"]/seg:>5.1f}/s · '
                  f'{hechos:,}/{len(filas):,}', flush=True)

    threading.Thread(target=reloj, daemon=True).start()
    try:
        with ThreadPoolExecutor(max_workers=hilos) as ex:
            list(ex.map(una, filas))
    finally:
        parar.set()

    seg = max(time.time() - t0, 1)
    print(f'\nSubidos: {n["subidos"]:,} ({n["bytes"]/1e9:.2f} GB) · '
          f'ya estaban: {n["ya_estaban"]:,} · sin archivo: {n["sin_archivo"]:,} · '
          f'errores: {n["errores"]:,} · {seg/60:.1f} min')


# ── FASE 2: cablear ──────────────────────────────────────────────────────
def cablear():
    """Escribe `pdf_url` en los puntos cuyo `registro` coincide.

    Se hace con `set_payload` filtrado por registro y no punto a punto: la
    colección tiene 71,655 puntos y muchos comparten registro —una tesis puede
    estar troceada—, así que filtrar es a la vez más rápido y más correcto que
    ir por id.
    """
    url = env('QDRANT_URL').rstrip('/')
    clave = env('QDRANT_API_KEY')
    filas = leer_manifiesto()
    print(f'{len(filas):,} tesis por cablear', flush=True)

    def pedir(ruta, cuerpo, metodo='POST'):
        req = urllib.request.Request(
            url + ruta, data=json.dumps(cuerpo).encode('utf-8'),
            headers={'api-key': clave, 'Content-Type': 'application/json'},
            method=metodo)
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read().decode('utf-8'))

    # EN PARALELO, PORQUE EN SERIE SON SEIS HORAS. Es una llamada por tesis
    # —cada una lleva su propia URL, así que no hay lote posible— y en serie
    # salían 3 por segundo: 63,172 tesis habrían tardado 5.8 horas. Qdrant
    # aguanta de sobra la concurrencia; el cuello era la latencia de ida y
    # vuelta, y eso se tapa con hilos.
    lock = threading.Lock()
    n = {'hechos': 0, 'errores': 0}
    t0 = time.time()

    def una(fila):
        reg = fila['registro']
        try:
            pedir(f'/collections/{COLECCION}/points/payload?wait=false', {
                'payload': {'pdf_url': url_de(reg)},
                'filter': {'must': [{'key': 'registro', 'match': {'value': reg}}]},
            })
            with lock:
                n['hechos'] += 1
        except Exception as e:
            with lock:
                n['errores'] += 1
                if n['errores'] <= 5:
                    print(f'  error con {reg}: {type(e).__name__}: {e}', flush=True)

    parar = threading.Event()

    def reloj():
        while not parar.wait(15):
            seg = max(time.time() - t0, 1)
            print(f'  {n["hechos"]:,}/{len(filas):,} cableadas · '
                  f'{n["hechos"]/seg:.0f}/s · errores {n["errores"]:,}', flush=True)

    threading.Thread(target=reloj, daemon=True).start()
    try:
        with ThreadPoolExecutor(max_workers=48) as ex:
            list(ex.map(una, filas))
    finally:
        parar.set()

    print(f'\nCableadas: {n["hechos"]:,} · errores: {n["errores"]:,} · '
          f'{(time.time()-t0)/60:.1f} min')
    print(f'Ejemplo: {url_de(filas[0]["registro"])}')


def main():
    hilos = int(sys.argv[sys.argv.index('--hilos') + 1]) if '--hilos' in sys.argv else HILOS
    hacer_subir = '--subir' in sys.argv or ('--cablear' not in sys.argv)
    hacer_cablear = '--cablear' in sys.argv or ('--subir' not in sys.argv)

    if hacer_subir:
        print('══ FASE 1 · subir al bucket ══')
        subir(hilos)
    if hacer_cablear:
        print('\n══ FASE 2 · cablear con la cita ══')
        cablear()


if __name__ == '__main__':
    main()
