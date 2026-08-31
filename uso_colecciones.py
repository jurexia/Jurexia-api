"""QUÉ COLECCIONES SE USAN DE VERDAD, contadas en producción.

David, 31-ago-2026: «jurisprudencia v2 no debería estar siendo utilizada porque
la reemplazamos por la v3, no creo que hayas hecho una búsqueda correcta. Hay,
por ejemplo, leyes estatales que nunca se pican en consulta. Si quieres
asegurarte, basta con implementar un mecanismo de logs».

Tiene razón y mi método era flojo. Yo busqué el NOMBRE de cada colección en el
código y conté 53 de 53 «en uso», pero que un nombre aparezca sólo prueba que
alguien lo escribió: no prueba que esa rama se ejecute, ni que un usuario pase
por ella en un día normal. Con eso concluí «no borres nada», que es una
recomendación cómoda y mal fundada.

Esto cuenta las llamadas REALES. Se envuelven los métodos del cliente de Qdrant
una sola vez, al arrancar, y cada uno apunta la colección que toca. Un solo
sitio: si lo pusiera en cada llamada, la que se me olvidara sería justo la que
decidiera un borrado.

NO GUARDA NADA DEL USUARIO: sólo el nombre de la colección y un contador. Ni la
consulta, ni el correo, ni el vector.

Se lee en `GET /admin/uso-colecciones`. Tras una semana de tráfico normal, lo
que tenga cero es candidato a borrar —y entonces sí, con datos—.
"""

from __future__ import annotations

import threading
import time

_LOCK = threading.Lock()
_CUENTA: dict[str, dict] = {}
_DESDE = time.time()
_METODOS = ("search", "query_points", "scroll", "retrieve", "count",
            "search_batch", "query_batch_points", "upsert", "delete")


def _apunta(coleccion: str, metodo: str) -> None:
    if not coleccion:
        return
    with _LOCK:
        c = _CUENTA.setdefault(str(coleccion), {"total": 0, "por_metodo": {}})
        c["total"] += 1
        c["por_metodo"][metodo] = c["por_metodo"].get(metodo, 0) + 1
        c["ultima"] = time.time()


def instrumentar(cliente) -> int:
    """Envuelve los métodos del cliente. Devuelve cuántos envolvió.

    Si algo falla se devuelve 0 y el cliente se queda intacto: contar el uso no
    puede ser motivo de que el API deje de responder.
    """
    envueltos = 0
    for nombre in _METODOS:
        try:
            original = getattr(cliente, nombre, None)
            if original is None or getattr(original, "_contado", False):
                continue

            def _envolver(orig, meth):
                async def _async(*a, **k):
                    _apunta(k.get("collection_name") or (a[0] if a else ""), meth)
                    return await orig(*a, **k)

                def _sync(*a, **k):
                    _apunta(k.get("collection_name") or (a[0] if a else ""), meth)
                    return orig(*a, **k)

                import inspect
                f = _async if inspect.iscoroutinefunction(orig) else _sync
                f._contado = True
                return f

            setattr(cliente, nombre, _envolver(original, nombre))
            envueltos += 1
        except Exception:
            continue
    return envueltos


def informe() -> dict:
    with _LOCK:
        datos = {k: dict(v) for k, v in _CUENTA.items()}
    horas = max(0.01, (time.time() - _DESDE) / 3600)
    usadas = sorted(datos.items(), key=lambda x: -x[1]["total"])
    return {
        "midiendo_desde_horas": round(horas, 2),
        "colecciones_tocadas": len(datos),
        "llamadas_totales": sum(v["total"] for v in datos.values()),
        "uso": [{"coleccion": k, **v} for k, v in usadas],
        "nota": ("Lo que no aparece aquí NO se ha tocado desde el último "
                 "reinicio. Con dos workers de gunicorn cada uno lleva su "
                 "cuenta: pide el informe varias veces y suma, o mira si el "
                 "total sube."),
    }
