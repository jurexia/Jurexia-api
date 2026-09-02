#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paso 2 de 3 — traer los PDF oficiales al disco externo.

POR QUÉ ESTE SCRIPT EXISTE (2-sep-2026)
---------------------------------------
El Semanario puso Incapsula delante de su API y ahora reta a las IP de centro
de datos. Medido: desde una conexión doméstica el mismo registro responde 200;
desde Vercel, Render y el runtime de borde, un 302 que redirige a sí mismo en
bucle o acaba en 403. Ningún servidor nuestro va a poder pedirle un PDF nunca
más. Lo que sí puede es esta máquina: se bajan una vez desde aquí, se suben a
nuestro bucket, y el visor deja de depender de un cortafuegos ajeno.

LA LECCIÓN CARA: NO ERA LA IP, ERA EL CLIENTE
---------------------------------------------
La primera versión usaba `httpx` y fracasó: 8 aciertos y 44 rechazos a
concurrencia 8, y con 2 conexiones fallaba el 90%. La conclusión parecía
obvia —«Incapsula sólo tolera una conexión por IP»— y era FALSA. Medido en
serie, en el mismo minuto y con las mismas cabeceras:

    urllib, 10 peticiones seguidas ....... 10 de 10
    httpx,  10 peticiones seguidas ....... 1 de 10, nueve 403

El cortafuegos toma la huella del CLIENTE —orden de cabeceras y handshake
TLS—, no del ritmo. Con `urllib` el paralelismo deja de ser un problema:

    urllib ·  4 hilos ...... 24 de 24 ·  3.9 PDF/s
    urllib ·  8 hilos ...... 24 de 24 ·  7.9 PDF/s
    urllib · 16 hilos ...... 40 de 40 · 12.4 PDF/s
    urllib · 32 hilos ...... 40 de 40 · 15.6 PDF/s · 8.8 MB/s

A 32 hilos las 71,655 tesis son ~77 minutos, no las 17 horas que prometía el
diagnóstico equivocado. De ahí en adelante el techo ya no es el servidor sino
el ancho de banda.

TRAMPA DEL ENDPOINT, heredada de la ruta del frontend: **responde 200 aunque
el registro no exista**, devolviendo un PDF truncado. Por eso cada archivo se
valida por cabecera `%PDF` y por tamaño mínimo antes de darlo por bueno.

    ./.venv/bin/python3 tesis_pdf_2_descargar.py
    ./.venv/bin/python3 tesis_pdf_2_descargar.py --hilos 16 --limite 200
"""
import hashlib, io, json, os, sys, threading, time
import urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = 'https://sjf2.scjn.gob.mx'
API = f'{BASE}/services/sjftesismicroservice/api/public/tesis'
# Dónde viven los PDF. Se movieron del USB al disco interno el 2-sep-2026:
# el Kingston se desmontó solo a mitad de la descarga —62,936 archivos
# dentro— y con él se fueron el proceso y el registro de progreso. Un
# trabajo de tres horas no puede colgar de un cable.
RAIZ = os.environ.get(
    'IUREXIA_TESIS_DIR',
    os.path.expanduser('~/Documents/KINGSTON/iurexia-tesis'))
PDFS = os.path.join(RAIZ, 'pdf')
MANIFIESTO = os.path.join(RAIZ, 'manifiesto.jsonl')

UA = ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 '
      '(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36')

MINIMO_BYTES = 20_000     # un PDF de tesis real ronda los 568 KB
HILOS = 32
REINTENTOS = 4

# Si de las últimas 200 peticiones falla más de este tanto por uno, algo
# cambió al otro lado y seguir insistiendo sólo empeora las cosas.
UMBRAL_ABORTO = 0.45

_lock = threading.Lock()
_cuenta = {'ok': 0, 'saltados': 0, 'invalidos': 0, 'rechazos': 0,
           'errores': 0, 'agotados': 0, 'bytes': 0, 'hechos': 0}
_ventana: list = []
_abortar = threading.Event()


def ruta_de(registro: str) -> str:
    # Repartidas por los tres primeros dígitos: 71,655 archivos en una sola
    # carpeta hacen lento hasta un `ls`.
    return os.path.join(PDFS, registro[:3], f'{registro}.pdf')


def ya_esta(registro: str) -> bool:
    try:
        return os.path.getsize(ruta_de(registro)) >= MINIMO_BYTES
    except OSError:
        return False


def anota(clave: str, bien: bool, octetos: int = 0):
    with _lock:
        _cuenta[clave] += 1
        _cuenta['hechos'] += 1
        _cuenta['bytes'] += octetos
        _ventana.append(bien)
        if len(_ventana) > 200:
            del _ventana[:-200]
        if len(_ventana) == 200 and (_ventana.count(False) / 200.0) > UMBRAL_ABORTO:
            _abortar.set()


def bajar(item: dict):
    reg = item['registro']
    if _abortar.is_set():
        return
    if ya_esta(reg):
        with _lock:
            _cuenta['saltados'] += 1
            _cuenta['hechos'] += 1
        return

    url = (f'{API}/reporte/{reg}?isSemanal=false&nameDocto=Tesis'
           f'&hostName={BASE}&soloParrafos=false')
    # El cortafuegos exige Referer Y User-Agent de navegador A LA VEZ: cada uno
    # por separado devuelve 403. Medido el 2-sep-2026.
    cab = {'Accept': '*/*', 'Accept-Language': 'es-MX,es;q=0.9',
           'Referer': f'{BASE}/detalle/tesis/{reg}', 'User-Agent': UA}

    for intento in range(REINTENTOS):
        if _abortar.is_set():
            return
        try:
            peticion = urllib.request.Request(url, headers=cab)
            with urllib.request.urlopen(peticion, timeout=60) as r:
                datos, estado = r.read(), r.status
        except urllib.error.HTTPError as e:
            estado, datos = e.code, b''
        except Exception:
            with _lock:
                _cuenta['errores'] += 1
            time.sleep(1.0 * (intento + 1))
            continue

        if estado in (403, 302, 429, 503):
            anota('rechazos', False)
            time.sleep(2 ** (intento + 1))        # 2, 4, 8, 16 s
            continue

        if estado != 200:
            anota('errores', False)
            return

        if not datos.startswith(b'%PDF') or len(datos) < MINIMO_BYTES:
            # 200 con basura = el registro no existe en el Semanario. No es un
            # fallo nuestro, así que cuenta como petición sana.
            anota('invalidos', True)
            return

        destino = ruta_de(reg)
        os.makedirs(os.path.dirname(destino), exist_ok=True)
        tmp = f'{destino}.parcial'
        with open(tmp, 'wb') as f:
            f.write(datos)
        os.replace(tmp, destino)

        linea = json.dumps({
            'registro': reg, 'bytes': len(datos),
            'sha1': hashlib.sha1(datos).hexdigest(),
            'rubro': item.get('rubro'), 'clave_tesis': item.get('clave_tesis'),
            'instancia': item.get('instancia'), 'epoca': item.get('epoca'),
            'tipo': item.get('tipo'), 'materia': item.get('materia'),
            'localizacion': item.get('localizacion'),
        }, ensure_ascii=False)
        with _lock:
            with io.open(MANIFIESTO, 'a', encoding='utf-8') as f:
                f.write(linea + '\n')
        anota('ok', True, len(datos))
        return

    anota('agotados', False)


def main():
    hilos = int(sys.argv[sys.argv.index('--hilos') + 1]) if '--hilos' in sys.argv else HILOS
    limite = int(sys.argv[sys.argv.index('--limite') + 1]) if '--limite' in sys.argv else None

    with io.open(os.path.join(RAIZ, 'registros.json'), encoding='utf-8') as f:
        universo = json.load(f)
    if limite:
        universo = universo[:limite]

    pendientes = [x for x in universo if not ya_esta(x['registro'])]
    print(f'{len(universo):,} en el censo · {len(pendientes):,} por bajar · '
          f'{hilos} hilos', flush=True)
    if not pendientes:
        print('Nada que hacer.')
        return

    t0 = time.time()
    parar_reloj = threading.Event()

    def reloj():
        while not parar_reloj.wait(20):
            with _lock:
                c = dict(_cuenta)
            seg = max(time.time() - t0, 1)
            ritmo = c['ok'] / seg
            faltan = (len(pendientes) - c['hechos']) / max(ritmo, 0.01) / 60
            print(f'  {c["ok"]:>6,}/{len(pendientes):,} · {c["bytes"]/1e9:>5.2f} GB · '
                  f'{ritmo:>5.1f}/s · {ritmo*0.568:>4.1f} MB/s · '
                  f'{c["invalidos"]:>4} inexistentes · {c["rechazos"]:>4} rechazos · '
                  f'faltan ~{faltan:>4.0f} min', flush=True)

    threading.Thread(target=reloj, daemon=True).start()
    try:
        with ThreadPoolExecutor(max_workers=hilos) as ex:
            list(ex.map(bajar, pendientes))
    finally:
        parar_reloj.set()

    seg = max(time.time() - t0, 1)
    print('\n' + '─' * 62)
    if _abortar.is_set():
        print('ABORTADO: más del 45% de las últimas 200 peticiones falló.')
        print('Algo cambió al otro lado. Relánzalo: continúa donde se quedó.')
    print(f'Descargados : {_cuenta["ok"]:,}  ({_cuenta["bytes"]/1e9:.2f} GB)')
    print(f'Ya estaban  : {_cuenta["saltados"]:,}')
    print(f'Inexistentes: {_cuenta["invalidos"]:,}  (200 con PDF truncado)')
    print(f'Rechazos    : {_cuenta["rechazos"]:,} · errores: {_cuenta["errores"]:,} · '
          f'agotados: {_cuenta["agotados"]:,}')
    print(f'Tiempo      : {seg/60:.1f} min · {_cuenta["ok"]/seg:.1f} PDF/s')


if __name__ == '__main__':
    main()
