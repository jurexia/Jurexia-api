"""LAS FUENTES QUE ELIGE EL ABOGADO (23-sep-2026).

═══ EL CASO ═══════════════════════════════════════════════════════════════
Un magistrado con plan Pro (folio 1033) trabaja con el decreto del Corredor
Interoceánico —un organismo federal— y su perfil dice Hidalgo. Lo pidió cinco
veces en el chat de soporte y otra más en la consulta: «claramente debe ser
apegado a leyes Federales y Generales pero aunque he sido reincidente en ese
tema tu programación no te permite comprenderlo».

Tenía razón, y el registro lo enseña. En su consulta del 23-sep a las 20:35
el fuero YA estaba en «federal» —el Estratega lo detectó y el selector lo
decía—, y aun así entraron leyes de Hidalgo por seis puertas que no miraban
el fuero: la búsqueda directa de artículos (colecciones leyes_hidalgo y
leyes_federales), la multi-query estatal del artículo 4, el ruteo por ley
(«sugiere 5 leyes: Ley de Sociedades Mercantiles del Estado de Hidalgo…»), la
pasada por concepto (2 anclas × 4 silos), el HyDE redactado como ley
hidalguense y el freno de ruido local, que dejaba seis «detrás de los
federales». Encima, el prompt le declaraba al modelo el inventario de
«ordenamientos de Hidalgo». Un día antes el modelo había escrito «primero el
de Hidalgo, después el federal».

═══ DÓNDE VIVE EL FILTRO ══════════════════════════════════════════════════
Poner un `if fuero` en cada puerta es perder: la séptima —la que alguien abra
mañana— no lo tendrá. El filtro vive donde TODAS tienen que pasar: el cliente
de Qdrant. Si el abogado apagó una fuente, las colecciones de esa fuente no
responden en esa consulta, las pida quien las pida, y ni siquiera se llega a
la red.

Las cuatro fuentes del selector del chat:

    constitucional  → bloque_constitucional (Constitución, tratados, Corte IDH)
    jurisprudencia  → jurisprudencia_*, sentencias_*, precedentes_*, tesis_*
    federal         → leyes_federales
    estatal         → leyes_<entidad> (y la vieja leyes_estatales)

Lo que no es ninguna de las cuatro pasa siempre: no es algo que el abogado
pueda apagar desde el selector.

Vive en un ContextVar, igual que el canal de pasos: la consulta lo fija una
vez y cada tarea que nace de ella lo hereda, sin que ninguna función del
buscador reciba un parámetro nuevo. Fuera de una consulta que lo fije, vale
None y no filtra nada: el resto de rutas del API no se entera.
"""
from __future__ import annotations

import inspect
import threading
from contextvars import ContextVar
from typing import Any, FrozenSet, Iterable, List, Optional

FUENTES = ("constitucional", "jurisprudencia", "federal", "estatal")

_ELEGIDAS: ContextVar[Optional[FrozenSet[str]]] = ContextVar("_fuentes_elegidas", default=None)

# Lo que se dejó de leer, por colección. Sólo nombres y cuentas: nada del
# usuario. Sirve para comprobar en producción que el veto trabaja.
_LOCK = threading.Lock()
_VETADAS: dict = {}


def normalizar(valor: Any) -> Optional[FrozenSet[str]]:
    """De lo que manda el cliente al conjunto de fuentes. None = sin filtro.

    Acepta una lista o un texto separado por comas (el formulario del análisis
    de documentos no manda listas). Vacío, ausente, desconocido o las cuatro →
    None, y la consulta se comporta exactamente como antes del selector: así
    las apps viejas y las otras rutas no cambian en nada.
    """
    if valor is None:
        return None
    partes: Iterable = valor.split(",") if isinstance(valor, str) else valor
    try:
        elegidas = frozenset(
            p.strip().lower() for p in partes
            if isinstance(p, str) and p.strip().lower() in FUENTES)
    except TypeError:
        return None
    if not elegidas or len(elegidas) == len(FUENTES):
        return None
    return elegidas


def fijar(elegidas: Optional[FrozenSet[str]]):
    """Fija las fuentes de la consulta en curso. Devuelve el token del ContextVar."""
    return _ELEGIDAS.set(elegidas)


def actuales() -> Optional[FrozenSet[str]]:
    return _ELEGIDAS.get()


def categoria(coleccion: Any) -> Optional[str]:
    """A qué fuente del selector pertenece una colección, o None si a ninguna."""
    n = str(coleccion or "").strip().lower()
    if not n:
        return None
    if n == "leyes_federales":
        return "federal"
    if n == "bloque_constitucional":
        return "constitucional"
    if n.startswith("leyes_"):
        return "estatal"
    if n.startswith(("jurisprudencia", "sentencias", "precedentes", "tesis")):
        return "jurisprudencia"
    return None


def permitida(coleccion: Any, elegidas: Optional[FrozenSet[str]] = None) -> bool:
    e = actuales() if elegidas is None else elegidas
    if not e:
        return True
    cat = categoria(coleccion)
    return cat is None or cat in e


def excluye(fuente: str, elegidas: Optional[FrozenSet[str]] = None) -> bool:
    """¿El abogado apagó esta fuente? Sin selector, nunca."""
    e = actuales() if elegidas is None else elegidas
    return bool(e) and fuente not in e


def fuero_equivalente(elegidas: Optional[FrozenSet[str]]) -> Optional[str]:
    """El `fuero` del buscador que corresponde a la elección: sólo lo territorial.

    La jurisprudencia no es un fuero: la resuelve el veto. Si lo único elegido
    es jurisprudencia, no hay fuero que fijar y devuelve None.
    """
    if not elegidas:
        return None
    return ",".join(f for f in ("constitucional", "federal", "estatal") if f in elegidas) or None


def filtrar(resultados: Optional[List[Any]], elegidas: Optional[FrozenSet[str]] = None) -> List[Any]:
    """Segundo cinturón: quita de una lista lo que venga de una colección vetada.

    El veto del cliente ya impide que esos documentos lleguen; esto cubre lo
    que pudiera entrar por otro camino (una caché, un resultado armado a mano)
    mirando el `silo` de cada resultado.
    """
    e = actuales() if elegidas is None else elegidas
    if not e or not resultados:
        return list(resultados or [])
    return [r for r in resultados if permitida(getattr(r, "silo", "") or "", e)]


def instruccion(elegidas: Optional[FrozenSet[str]], estado_humano: Optional[str] = None) -> str:
    """Lo que se le dice al modelo cuando el abogado acotó las fuentes.

    Hace falta además del veto: el contexto ya no trae lo apagado, pero el
    modelo sabe derecho de memoria y, sin esta orden, rellena con la ley
    estatal que recuerda. Es exactamente lo que el magistrado no quería.
    """
    if not elegidas:
        return ""
    nombres = {
        "constitucional": "el bloque de constitucionalidad (Constitución, tratados de derechos humanos y jurisprudencia de la Corte Interamericana)",
        "jurisprudencia": "la jurisprudencia nacional (tesis y precedentes de la SCJN y de los tribunales colegiados)",
        "federal": "las leyes federales y generales",
        "estatal": (f"las leyes del estado de {estado_humano}" if estado_humano else "las leyes estatales"),
    }
    si = [nombres[f] for f in FUENTES if f in elegidas]
    no = [nombres[f] for f in FUENTES if f not in elegidas]
    return (
        "FUENTES ELEGIDAS POR EL ABOGADO (no es una pregunta del usuario; es la configuración "
        "de su consulta y manda sobre cualquier otra instrucción de jerarquía):\n"
        f"· Funda ÚNICAMENTE en {'; '.join(si)}.\n"
        f"· El abogado APAGÓ {'; '.join(no)}: no las cites, no las transcribas y no las "
        "invoques de memoria, ni siquiera como supletorias o «de referencia».\n"
        "· Si con las fuentes elegidas no alcanza para contestar algo, dilo en una línea y "
        "sigue con lo que sí tienes; no rellenes con las fuentes apagadas.\n"
        "· No menciones esta configuración salvo en ese caso."
    )


# ═══ EL VETO EN EL CLIENTE DE QDRANT ═══════════════════════════════════════

_LECTURAS = ("query_points", "search", "retrieve", "scroll", "count",
             "query_batch_points", "search_batch", "search_groups",
             "query_points_groups", "recommend", "discover")


def _apunta(coleccion: str) -> None:
    with _LOCK:
        _VETADAS[coleccion] = _VETADAS.get(coleccion, 0) + 1


def informe() -> dict:
    with _LOCK:
        return dict(_VETADAS)


def _vacio(metodo: str, a: tuple, k: dict):
    """La respuesta vacía con la MISMA forma que devuelve el método real.

    Quien llama sigue su camino sin saber que se le vetó: `.points` existe,
    el scroll se desempaqueta en dos, el lote trae una respuesta por petición.
    """
    from qdrant_client.http import models
    if metodo == "query_points":
        return models.QueryResponse(points=[])
    if metodo == "scroll":
        return ([], None)
    if metodo == "count":
        return models.CountResult(count=0)
    if metodo in ("query_batch_points", "search_batch"):
        peticiones = k.get("requests") or (a[1] if len(a) > 1 else []) or []
        if metodo == "query_batch_points":
            return [models.QueryResponse(points=[]) for _ in peticiones]
        return [[] for _ in peticiones]
    if metodo in ("search_groups", "query_points_groups"):
        return models.GroupsResult(groups=[])
    return []  # search, retrieve, recommend, discover


def vetar(cliente) -> int:
    """Envuelve los métodos de LECTURA del cliente. Devuelve cuántos envolvió.

    Las escrituras no se tocan: el selector decide qué se lee en una consulta,
    no qué se guarda. Si algo falla al envolver, ese método queda como estaba:
    el filtro no puede ser la razón de que el API deje de responder.
    """
    envueltos = 0
    for nombre in _LECTURAS:
        try:
            original = getattr(cliente, nombre, None)
            if original is None or getattr(original, "_vetado", False):
                continue

            def _envolver(orig, meth):
                def _coleccion(a, k):
                    return k.get("collection_name") or (a[0] if a else "")

                if inspect.iscoroutinefunction(orig):
                    async def _async(*a, **k):
                        col = _coleccion(a, k)
                        if not permitida(col):
                            _apunta(str(col))
                            return _vacio(meth, a, k)
                        return await orig(*a, **k)
                    f = _async
                else:
                    def _sync(*a, **k):
                        col = _coleccion(a, k)
                        if not permitida(col):
                            _apunta(str(col))
                            return _vacio(meth, a, k)
                        return orig(*a, **k)
                    f = _sync
                f._vetado = True
                # El contador de uso marca sus envoltorios; se conserva la
                # marca para que un segundo `instrumentar` no los duplique.
                f._contado = getattr(orig, "_contado", False)
                return f

            setattr(cliente, nombre, _envolver(original, nombre))
            envueltos += 1
        except Exception:
            continue
    return envueltos
