"""EL ANÁLISIS NO SE QUEDA CALLADO MIENTRAS EL MODELO PIENSA.   (29-oct-2026)

/analyze-document latía cada 5 s sólo mientras leía el documento. Después,
al abrir el modelo y hasta su primer token, no mandaba NADA. Platinum escribe
con gemini-3.8-flash: en 308 análisis del 22 al 29-oct ese silencio fue de 20 s
de mediana, 43.9 s en el percentil 90 y hasta 66 s (83 pasaron de 30 s), a 9 s
de los 75 con que la pantalla se rendía («Se perdió el contacto con el
servidor»). Las redes móviles, los proxies de oficina y los balanceadores
cierran la conexión callada —justo por eso el /chat manda su <!--PING-->—, y
lo mismo pasaba al abrir la continuación cuando Gemini corta por RECITATION
(41 veces en esos ocho días).

Aquí el flujo del modelo se ABRE y se RECORRE en una tarea aparte, y a quien
lo consume se le entrega cada trozo o, si pasan `cada` segundos sin nada, el
centinela LATIDO, que la ruta convierte en un comentario SSE (`: latido`): la
pantalla no lo pinta pero rearma su reloj, y la red ve tráfico.

UNA SOLA TAREA PARA ABRIR Y LEER, a propósito: el SDK de Gemini lee con
aiohttp, cuyos plazos pertenecen a la tarea que hizo la petición; abrir en una
y leer en otra es buscarse un corte raro. El consumidor sólo espera la cola.

Si el consumidor se va (el navegador cerró), la tarea se cancela y con ella la
petición al modelo: no se sigue pagando un análisis que nadie va a leer.
"""
from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Awaitable, Callable

# El centinela: «pasó el plazo sin trozo». Se compara por identidad.
LATIDO = object()
_FIN = object()


async def con_latido(abrir: Callable[[], Awaitable[Any]], cada: float = 5.0) -> AsyncIterator[Any]:
    """Abre el flujo con `abrir()` y entrega sus trozos, intercalando LATIDO
    cada `cada` segundos de silencio (también antes del primero).

    Los errores de `abrir()` o de la lectura suben tal cual en el consumidor,
    en el punto donde los esperaba: el repliegue al abrir y el «¿hubo texto?»
    de la ruta no cambian."""
    cola: asyncio.Queue = asyncio.Queue()

    async def _bomba() -> None:
        try:
            flujo = await abrir()
            async for trozo in flujo:
                await cola.put((True, trozo))
            await cola.put((True, _FIN))
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 — se entrega al consumidor
            await cola.put((False, e))

    tarea = asyncio.create_task(_bomba())
    # La espera vive entre latidos: no se cancela un `get` que pudo haber
    # sacado ya un trozo de la cola (se perdería texto).
    espera = None
    try:
        while True:
            if espera is None:
                espera = asyncio.ensure_future(cola.get())
            hecho, _ = await asyncio.wait({espera}, timeout=cada)
            if not hecho:
                yield LATIDO
                continue
            ok, valor = espera.result()
            espera = None
            if not ok:
                raise valor
            if valor is _FIN:
                return
            yield valor
    finally:
        if espera is not None and not espera.done():
            espera.cancel()
        if not tarea.done():
            tarea.cancel()
