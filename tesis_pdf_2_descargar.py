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
más.

Lo que sí puede es esta máquina. Así que se bajan una vez desde aquí, se
suben a nuestro bucket, y el visor deja de depender del humor de un
cortafuegos ajeno — igual que ya se hizo con las sentencias de TCC.

LA PRUDENCIA NO ES OPCIONAL
---------------------------
Esta IP doméstica es la única puerta que queda abierta. Si se la satura y
Incapsula la marca, no hay plan C. Por eso la concurrencia es ADAPTATIVA y
asimétrica: sube despacio cuando todo va bien y se desploma al primer indicio
de rechazo. Y si el rechazo se sostiene, el script para solo en lugar de
insistir: es preferible terminar mañana que quedarse sin fuente hoy.

TRAMPA DEL ENDPOINT, heredada de la ruta del frontend: **responde 200 aunque
el registro no exista**, devolviendo un PDF truncado. Por eso cada archivo se
valida por cabecera `%PDF` y por tamaño mínimo antes de darlo por bueno.

    ./.venv/bin/python3 tesis_pdf_2_descargar.py            # todo
    ./.venv/bin/python3 tesis_pdf_2_descargar.py --limite 50  # prueba corta
"""
import asyncio, hashlib, io, json, os, sys, time
import httpx

BASE = 'https://sjf2.scjn.gob.mx'
API = f'{BASE}/services/sjftesismicroservice/api/public/tesis'
RAIZ = '/Volumes/KINGSTON/iurexia-tesis'
PDFS = os.path.join(RAIZ, 'pdf')
MANIFIESTO = os.path.join(RAIZ, 'manifiesto.jsonl')

UA = ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 '
      '(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36')

# Un PDF de tesis real pesa ~570 KB; los truncados que devuelve el endpoint
# para un registro inexistente rondan los pocos KB.
MINIMO_BYTES = 20_000

# Medido el 2-sep-2026: a concurrencia 8 el Semanario devolvió 44 rechazos por
# 8 aciertos. No estrangula por conexiones simultáneas sino por PETICIONES POR
# SEGUNDO, así que la palanca de verdad es el intervalo, no la concurrencia.
CONC_INICIAL, CONC_MAXIMA, CONC_MINIMA = 3, 6, 1
RECHAZOS_PARA_PARAR = 60      # rechazos seguidos = Incapsula nos marcó

# Segundos entre el arranque de una petición y la siguiente. Se ajusta solo:
# baja despacio mientras todo va bien y sube de golpe al primer rechazo.
INTERVALO_INICIAL, INTERVALO_MINIMO, INTERVALO_MAXIMO = 0.45, 0.15, 4.0


def ruta_de(registro: str) -> str:
    # Repartidas por los tres primeros dígitos: 71,655 archivos en una sola
    # carpeta hacen lento hasta un `ls`.
    return os.path.join(PDFS, registro[:3], f'{registro}.pdf')


def ya_esta(registro: str) -> bool:
    p = ruta_de(registro)
    try:
        return os.path.getsize(p) >= MINIMO_BYTES
    except OSError:
        return False


class Ritmo:
    """El ritmo de las peticiones: sube despacio, baja de golpe.

    La asimetría es deliberada. Recuperar velocidad perdida cuesta minutos;
    recuperar una IP marcada por Incapsula puede costar el proyecto entero,
    porque esta conexión doméstica es la única que el Semanario todavía
    atiende.
    """

    def __init__(self, conc=CONC_INICIAL, intervalo=INTERVALO_INICIAL):
        self.n = conc
        self.sem = asyncio.Semaphore(conc)
        self.intervalo = intervalo
        self.siguiente_hueco = 0.0
        self.buenas_seguidas = 0
        self.rechazos_seguidos = 0
        self.parar = False

    async def esperar_turno(self):
        """Reparte las salidas en el tiempo, que es lo que de verdad mira el
        cortafuegos: no cuántas conexiones hay abiertas, sino cuántas
        peticiones llegan por segundo."""
        ahora = asyncio.get_event_loop().time()
        salida = max(ahora, self.siguiente_hueco)
        self.siguiente_hueco = salida + self.intervalo
        if salida > ahora:
            await asyncio.sleep(salida - ahora)

    def bien(self):
        self.rechazos_seguidos = 0
        self.buenas_seguidas += 1
        if self.buenas_seguidas >= 80:
            self.buenas_seguidas = 0
            self.intervalo = max(INTERVALO_MINIMO, self.intervalo * 0.85)
            if self.n < CONC_MAXIMA:
                self.n += 1
                self.sem.release()

    def rechazado(self):
        self.buenas_seguidas = 0
        self.rechazos_seguidos += 1
        if self.rechazos_seguidos >= RECHAZOS_PARA_PARAR:
            self.parar = True
        self.intervalo = min(INTERVALO_MAXIMO, max(self.intervalo, 0.2) * 2.0)
        objetivo = max(CONC_MINIMA, self.n // 2)
        while self.n > objetivo:
            self.n -= 1
            asyncio.create_task(self.sem.acquire())


async def bajar(cli: httpx.AsyncClient, item: dict, ritmo: Ritmo, cuenta: dict):
    reg = item['registro']
    if ya_esta(reg):
        cuenta['saltados'] += 1
        return

    url = (f'{API}/reporte/{reg}?isSemanal=false&nameDocto=Tesis'
           f'&hostName={BASE}&soloParrafos=false')
    cab = {'Accept': '*/*', 'Accept-Language': 'es-MX,es;q=0.9',
           'Referer': f'{BASE}/detalle/tesis/{reg}', 'User-Agent': UA}

    for intento in range(4):
        if ritmo.parar:
            return
        async with ritmo.sem:
            await ritmo.esperar_turno()
            try:
                r = await cli.get(url, headers=cab)
            except Exception as e:
                await asyncio.sleep(1.5 * (intento + 1))
                cuenta['errores_red'] += 1
                continue

        if r.status_code in (403, 302, 429, 503):
            ritmo.rechazado()
            cuenta['rechazos'] += 1
            # Espera creciente: 3, 9, 27 segundos.
            await asyncio.sleep(3 ** (intento + 1))
            continue

        if r.status_code != 200:
            cuenta['otros'] += 1
            return

        datos = r.content
        if not datos.startswith(b'%PDF') or len(datos) < MINIMO_BYTES:
            # El endpoint contesta 200 con basura cuando el registro no existe.
            cuenta['invalidos'] += 1
            ritmo.bien()
            return

        destino = ruta_de(reg)
        os.makedirs(os.path.dirname(destino), exist_ok=True)
        tmp = destino + '.parcial'
        with open(tmp, 'wb') as f:
            f.write(datos)
        os.replace(tmp, destino)

        with io.open(MANIFIESTO, 'a', encoding='utf-8') as f:
            f.write(json.dumps({
                'registro': reg,
                'bytes': len(datos),
                'sha1': hashlib.sha1(datos).hexdigest(),
                'rubro': item.get('rubro'),
                'clave_tesis': item.get('clave_tesis'),
                'instancia': item.get('instancia'),
                'epoca': item.get('epoca'),
                'tipo': item.get('tipo'),
                'materia': item.get('materia'),
                'localizacion': item.get('localizacion'),
            }, ensure_ascii=False) + '\n')

        cuenta['ok'] += 1
        cuenta['bytes'] += len(datos)
        ritmo.bien()
        return

    cuenta['agotados'] += 1


async def main():
    limite = None
    if '--limite' in sys.argv:
        limite = int(sys.argv[sys.argv.index('--limite') + 1])

    with io.open(os.path.join(RAIZ, 'registros.json'), encoding='utf-8') as f:
        universo = json.load(f)
    if limite:
        universo = universo[:limite]

    pendientes = [x for x in universo if not ya_esta(x['registro'])]
    print(f'{len(universo):,} en el censo · {len(pendientes):,} por bajar', flush=True)
    if not pendientes:
        print('Nada que hacer.')
        return

    conc = int(sys.argv[sys.argv.index('--conc') + 1]) if '--conc' in sys.argv else CONC_INICIAL
    intervalo = (float(sys.argv[sys.argv.index('--intervalo') + 1])
                 if '--intervalo' in sys.argv else INTERVALO_INICIAL)
    ritmo = Ritmo(conc, intervalo)
    print(f'ritmo inicial: concurrencia {conc}, un arranque cada {intervalo:.2f} s', flush=True)
    cuenta = {'ok': 0, 'saltados': 0, 'invalidos': 0, 'rechazos': 0,
              'errores_red': 0, 'otros': 0, 'agotados': 0, 'bytes': 0}
    t0 = time.time()

    limites = httpx.Limits(max_connections=CONC_MAXIMA + 4,
                           max_keepalive_connections=CONC_MAXIMA)
    async with httpx.AsyncClient(http2=False, timeout=45.0, limits=limites,
                                 follow_redirects=False) as cli:
        tareas = set()
        for i, item in enumerate(pendientes, 1):
            if ritmo.parar:
                break
            t = asyncio.create_task(bajar(cli, item, ritmo, cuenta))
            tareas.add(t)
            t.add_done_callback(tareas.discard)

            # No se crean 71,655 tareas de golpe: se mantiene una ventana.
            while len(tareas) >= ritmo.n * 3:
                await asyncio.sleep(0.02)

            if i % 250 == 0:
                seg = time.time() - t0
                print(f'  {cuenta["ok"]:>6,} ok · {cuenta["invalidos"]:>4} inexistentes · '
                      f'{cuenta["rechazos"]:>4} rechazos · conc {ritmo.n:>2} · '
                      f'{cuenta["bytes"]/1e9:>5.2f} GB · '
                      f'{cuenta["ok"]/max(seg,1):>5.1f}/s · int {ritmo.intervalo:.2f}s · '
                      f'{seg/60:>5.1f} min', flush=True)

        if tareas:
            await asyncio.gather(*tareas, return_exceptions=True)

    seg = time.time() - t0
    print('\n' + '─' * 60)
    if ritmo.parar:
        print('DETENIDO: demasiados rechazos seguidos. Incapsula nos está frenando.')
        print('Deja pasar un rato y vuelve a lanzarlo: continúa donde se quedó.')
    print(f'Descargados : {cuenta["ok"]:,}  ({cuenta["bytes"]/1e9:.2f} GB)')
    print(f'Ya estaban  : {cuenta["saltados"]:,}')
    print(f'Inexistentes: {cuenta["invalidos"]:,}  (200 con PDF truncado)')
    print(f'Rechazos    : {cuenta["rechazos"]:,} · red: {cuenta["errores_red"]:,} · '
          f'agotados: {cuenta["agotados"]:,}')
    print(f'Tiempo      : {seg/60:.1f} min · {cuenta["ok"]/max(seg,1):.1f} PDF/s')


if __name__ == '__main__':
    asyncio.run(main())
