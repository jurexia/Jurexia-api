"""LA LECTURA OCR DE UN PDF SE PAGA UNA VEZ: CACHÉ POR HUELLA.   (24-sep-2026)

POR QUÉ EXISTE
--------------
Septiembre en Azure: 40.19 USD, y 40.18 son un solo medidor, «S0 Read Pages»
—el OCR `prebuilt-read`, 1.50 USD por cada mil páginas—. No fue el uso de todos
los días (0.8 a 1.3 USD, lo que cuentan los registros de la API): fueron dos
picos, el 1-2 y el 9-11 de septiembre, unas 12,400 y 7,300 páginas, justo los
días en que el taller se probaba una y otra vez contra los mismos expedientes.
Cada corrida mandaba el expediente entero a Azure y lo volvía a pagar: nada
recordaba que ese mismo PDF ya se había leído.

QUÉ HACE
--------
La llave es el SHA-256 de los bytes del PDF. El mismo archivo da la misma
huella entre por donde entre —chat, taller, jurimetría, SISE— y la suba quien
la suba: sólo acierta quien ya tiene ese documento en la mano, así que no se
filtra nada entre usuarios. Se guarda comprimido en el cubo privado
`expedientes`, en `ocr-cache/v1/<huella>.json.gz`.

  · Sólo se guarda la lectura COMPLETA de Azure. La de Gemini no: es el
    respaldo de cuando Azure falló, y guardarla congelaría el peor texto y le
    quitaría a Azure la siguiente oportunidad.
  · Caduca a los OCR_CACHE_DIAS días (14 si no se dice). Es texto de
    expedientes y las constancias se sueltan al terminar el proyecto: esta
    copia no debe quedarse a vivir. Lo caducado no se lee, y una purga
    oportunista —una vez cada seis horas por proceso— lo borra del cubo.
  · Nada de esto puede tumbar una lectura. Si el almacén no responde, se lee
    como antes y se paga como antes.
"""
import asyncio
import gzip
import hashlib
import json
import os
import time
from datetime import datetime
from typing import Optional

CUBO = "expedientes"                       # privado, sin tope de tamaño ni de tipo
PREFIJO = "ocr-cache/v1"
# El motor va dentro de la ficha: si mañana se cambia de modelo o de versión de
# la API, lo guardado con el anterior deja de valer solo, sin purgar a mano.
MOTOR = "azure-prebuilt-read@2024-11-30"
DIAS = float(os.getenv("OCR_CACHE_DIAS", "14"))
_PURGA_CADA_S = 6 * 3600
_ultima_purga = 0.0
# Las subidas corren en segundo plano; sin una referencia viva, el recolector
# puede llevarse la tarea a media subida.
_TAREAS: set = set()


def huella(contenido: bytes) -> str:
    return hashlib.sha256(contenido).hexdigest()


def ruta(h: str) -> str:
    return f"{PREFIJO}/{h}.json.gz"


def empaquetar(texto: str, paginas: int, ahora: Optional[float] = None) -> bytes:
    ficha = {"v": 1, "motor": MOTOR, "paginas": int(paginas or 0),
             "creado": time.time() if ahora is None else ahora, "texto": texto}
    return gzip.compress(json.dumps(ficha, ensure_ascii=False).encode("utf-8"), compresslevel=6)


def desempaquetar(datos: bytes, ahora: Optional[float] = None,
                  dias: Optional[float] = None) -> Optional[dict]:
    """La ficha guardada, o None si no sirve: rota, de otro motor, vacía o caducada."""
    try:
        ficha = json.loads(gzip.decompress(datos).decode("utf-8"))
    except Exception:
        return None
    if not isinstance(ficha, dict) or ficha.get("v") != 1 or ficha.get("motor") != MOTOR:
        return None
    if not (ficha.get("texto") or "").strip():
        return None
    edad = (time.time() if ahora is None else ahora) - float(ficha.get("creado") or 0)
    # Una fecha del futuro es un reloj roto, no una lectura fresca.
    if edad > (DIAS if dias is None else dias) * 86400 or edad < -3600:
        return None
    return ficha


def _es_no_existe(e: Exception) -> bool:
    t = str(e).lower()
    return "not found" in t or "not_found" in t or "404" in t


async def leer(cliente, h: str) -> Optional[str]:
    """El texto ya leído de ese PDF, o None. Nunca lanza."""
    if cliente is None or not h:
        return None
    try:
        datos = await asyncio.to_thread(lambda: cliente.storage.from_(CUBO).download(ruta(h)))
    except Exception as e:
        # Que no exista es lo normal —es la primera vez—; cualquier otra cosa
        # se anota, pero la lectura sigue por Azure como siempre.
        if not _es_no_existe(e):
            print(f"   ⚠️ OCR caché: no se pudo consultar ({type(e).__name__})")
        return None
    ficha = await asyncio.to_thread(desempaquetar, datos)
    return ficha["texto"] if ficha else None


def guardar_en_segundo_plano(cliente, h: str, texto: str, paginas: int) -> None:
    """Sube la lectura sin hacer esperar al abogado que la pidió."""
    if cliente is None or not h or not (texto or "").strip():
        return
    tarea = asyncio.get_running_loop().create_task(_guardar(cliente, h, texto, paginas))
    _TAREAS.add(tarea)
    tarea.add_done_callback(_TAREAS.discard)


async def _guardar(cliente, h: str, texto: str, paginas: int) -> None:
    try:
        datos = await asyncio.to_thread(empaquetar, texto, paginas)
        await asyncio.to_thread(lambda: cliente.storage.from_(CUBO).upload(
            ruta(h), datos, {"content-type": "application/gzip", "upsert": "true"}))
        print(f"   💾 OCR caché: guardada la lectura de {paginas} pág "
              f"({len(datos) // 1024} KB, huella {h[:12]})")
    except Exception as e:
        print(f"   ⚠️ OCR caché: no se pudo guardar ({type(e).__name__})")
    await _purgar_si_toca(cliente)


async def _purgar_si_toca(cliente) -> None:
    global _ultima_purga
    if time.time() - _ultima_purga < _PURGA_CADA_S:
        return
    _ultima_purga = time.time()
    try:
        n = await asyncio.to_thread(purgar, cliente)
        if n:
            print(f"   🧹 OCR caché: {n} lecturas caducadas borradas del cubo")
    except Exception as e:
        print(f"   ⚠️ OCR caché: la purga falló ({type(e).__name__})")


def _segundos(iso) -> Optional[float]:
    try:
        return datetime.fromisoformat(str(iso).replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def purgar(cliente, dias: Optional[float] = None, ahora: Optional[float] = None,
           tope: int = 1000) -> int:
    """Borra del cubo las lecturas caducadas, de la más vieja a la más nueva.

    Se ordena por `updated_at` porque una lectura caducada que se vuelve a
    pedir se sobrescribe en la misma ruta: la fecha que manda es la de la
    última escritura, no la de la primera.
    """
    limite = (time.time() if ahora is None else ahora) - (DIAS if dias is None else dias) * 86400
    cubo = cliente.storage.from_(CUBO)
    objetos = cubo.list(PREFIJO, {"limit": tope, "offset": 0,
                                  "sortBy": {"column": "updated_at", "order": "asc"}}) or []
    viejos = []
    for o in objetos:
        nombre = o.get("name") or ""
        if not nombre.endswith(".json.gz"):
            continue
        cuando = _segundos(o.get("updated_at") or o.get("created_at"))
        if cuando is None:
            continue
        if cuando >= limite:
            break                       # vienen ordenados: lo que sigue es más nuevo
        viejos.append(f"{PREFIJO}/{nombre}")
    for i in range(0, len(viejos), 100):
        cubo.remove(viejos[i:i + 100])
    return len(viejos)
