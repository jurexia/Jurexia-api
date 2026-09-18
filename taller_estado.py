"""Lo que una fase del taller deja para la siguiente, y cómo vuelve entero.

POR QUÉ EXISTE. Revisión fiscal 2/2026, 17-sep-2026, medido en los registros de
producción minuto a minuto. El proyecto tardaba 16 minutos y una parte era
trabajo hecho DOS y TRES veces:

  - La consulta del acervo —nueve preguntas al modelo y ocho reordenaciones,
    unos 30 s— corría en /taller/consultar, OTRA VEZ en /taller/proponer y OTRA
    en /taller/resolver. El acervo se guardaba en la fila (`estado.material`)
    pero al rehidratar la sesión nadie lo reponía, así que el worker que no lo
    había consultado veía «material no está» y volvía a consultar. Con dos
    workers y sin afinidad, eso es casi siempre. Y cada consulta elige tesis
    distintas —la reordenación es del modelo—, así que la propuesta se hacía
    sobre un acervo y el estudio sobre otro.

  - El artículo 150 del Reglamento Interior del IMSS se traía de internet en
    la propuesta (30 s), se guardaba con el material… y el resolver, que había
    perdido el material, lo volvía a traer (otros 30 s).

  - El contraste de la propuesta —una llamada con razonamiento alto, 55 a
    125 s— sólo necesita lo que el adelanto ya produjo: los planteamientos y
    los dos resúmenes. Esperaba en serie dentro de /taller/proponer, cuando
    podía haberse calculado mientras el secretario consultaba el acervo.

Aquí viven las piezas puras de esas dos correcciones: el acervo que va a la
fila COMPLETO y vuelve como el mismo `Material` —con sondeo, principios, sede
del acto, cuaderno y entidad, que antes se perdían—, y la huella y la espera
del contraste adelantado. Nada de esto toca la base: `main.py` pone las
lecturas y escrituras alrededor.

Regla de la casa ([[estado-entre-workers]]): lo que hace falta en una petición
posterior va a la fila y se repone al rescatar la sesión. La memoria del
proceso es caché, no verdad.
"""
from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import time

# Tope de elementos por lista al guardar. No recorta nada real —el material
# lleva 80 tesis como mucho y un sondeo trae 40 razonados—; es un freno por si
# un día una lista crece sin medida y la fila deja de caber.
_TOPE_LISTA = 200


def _lista(xs, n: int = _TOPE_LISTA) -> list:
    return [x for x in (xs or [])][:n]


def sondeo_ligero(s) -> dict | None:
    """El `fase_precedente.Sondeo` como dict de JSON puro. None si no hay."""
    if s is None:
        return None
    try:
        d = dataclasses.asdict(s) if dataclasses.is_dataclass(s) else dict(vars(s))
    except Exception:
        return None
    # Ida y vuelta por JSON: lo que no sea JSON se convierte a texto en vez de
    # tumbar el guardado del material entero.
    try:
        return json.loads(json.dumps(d, ensure_ascii=False, default=str))
    except Exception:
        return None


def sondeo_rehidratado(d):
    """El dict de vuelta a `Sondeo`, con sus métodos. None si no se puede."""
    if not isinstance(d, dict) or not d:
        return None
    try:
        import fase_precedente as _fp
        campos = {f.name for f in dataclasses.fields(_fp.Sondeo)}
        return _fp.Sondeo(**{k: v for k, v in d.items() if k in campos})
    except Exception:
        return None


def material_ligero(m) -> dict:
    """El acervo tal como va a la fila. `completo=True` marca que lleva TODO lo
    que los prompts leen; sin esa marca el rescate de la sesión no lo repone y
    se vuelve a consultar, que es lo que pasaba con las filas antiguas."""
    def _dicts(xs, n):
        return [x for x in (xs or []) if isinstance(x, dict)][:n]
    return {
        "tesis": _dicts(getattr(m, "tesis", []), 80),
        "normas": _dicts(getattr(m, "normas", []), 80),
        "convencional": _dicts(getattr(m, "convencional", []), 24),
        "materia": str(getattr(m, "materia", "") or ""),
        "tipo_asunto": str(getattr(m, "tipo_asunto", "") or ""),
        # EL ESPEJO VIAJA POR LA FILA, NO POR LA MEMORIA. Son seis filas de siete
        # campos por planteamiento: kilobyte y medio.
        "espejo": _dicts(getattr(m, "espejo", []), 8),
        # LO QUE FALTABA Y OBLIGABA A CONSULTAR OTRA VEZ: sin el sondeo, la
        # propuesta perdía la corriente del acervo y el estudio sus moldes y
        # objeciones; sin los principios, el estudio no sabía con qué nociones
        # razona el circuito; sin sede y cuaderno, la verja de la suspensión
        # se quedaba ciega.
        "principios": _lista(getattr(m, "principios", []), 24),
        "sede_del_acto": str(getattr(m, "sede_del_acto", "") or ""),
        "cuaderno": str(getattr(m, "cuaderno", "") or ""),
        "entidad": str(getattr(m, "entidad", "") or ""),
        "preceptos_de_internet": _lista(getattr(m, "preceptos_de_internet", []), 24),
        "sondeo": sondeo_ligero(getattr(m, "sondeo", None)),
        "completo": True,
    }


def material_rehidratado(d: dict):
    """Un `Material` con lo guardado. None si no hay nada aprovechable."""
    if not isinstance(d, dict) or not (d.get("tesis") or d.get("normas")):
        return None
    import fase6_estudio as _f6m
    m = _f6m.Material()
    m.tesis = list(d.get("tesis") or [])
    m.normas = list(d.get("normas") or [])
    m.convencional = list(d.get("convencional") or [])
    m.materia = str(d.get("materia") or "")
    m.tipo_asunto = str(d.get("tipo_asunto") or "amparo_directo")
    m.espejo = [x for x in (d.get("espejo") or []) if isinstance(x, dict)]
    m.principios = list(d.get("principios") or [])
    m.sede_del_acto = str(d.get("sede_del_acto") or "")
    m.cuaderno = str(d.get("cuaderno") or "")
    m.entidad = str(d.get("entidad") or "")
    m.preceptos_de_internet = list(d.get("preceptos_de_internet") or [])
    m.sondeo = sondeo_rehidratado(d.get("sondeo"))
    return m


def esta_completo(d) -> bool:
    return isinstance(d, dict) and bool(d.get("completo")) and bool(
        d.get("tesis") or d.get("normas"))


# ═══ EL CONTRASTE ADELANTADO ═════════════════════════════════════════════════

def problemas_de(r) -> list:
    """Los planteamientos tal como los recibe la propuesta. UNA sola forma de
    construirlos, porque la huella se calcula sobre ellos."""
    f = getattr(r, "fases", None)
    problemas = [p if isinstance(p, dict) else {"pregunta": str(p)}
                 for p in (getattr(f, "problemas", None) or [])]
    if not problemas and getattr(f, "problema_global", ""):
        problemas = [{"pregunta": f.problema_global}]
    return problemas


def entradas_contraste(r) -> tuple:
    """(problemas, resumen_acto, resumen_conceptos, es_recurso): lo ÚNICO que
    el contraste lee. Ni el acervo ni el contexto del secretario entran, y por
    eso puede calcularse antes de consultar."""
    f = getattr(r, "fases", None)
    e = getattr(r, "encargo", None)
    return (problemas_de(r),
            "\n".join((f.parrafos_acto() if f else []) or []),
            "\n".join((f.parrafos_conceptos() if f else []) or []),
            bool(e is not None and getattr(e, "es_recurso", False)))


def huella_contraste(r) -> str:
    """Identifica el adelanto del que sale un contraste. Si el secretario
    rehace el adelanto o cambia un planteamiento, la huella cambia y el
    contraste guardado no se usa."""
    base = json.dumps(list(entradas_contraste(r)), ensure_ascii=False,
                      sort_keys=True, default=str)
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]


# LAS TAREAS SUELTAS LATEN. Mientras corren, rescriben su marca cada 45 s con
# `latido`; sin latido en este tiempo, el worker que las corría ya no está —un
# despliegue lo mata con SIGTERM y nadie escribe «fallo»— y se hace el trabajo
# aquí. Medido en producción (18-sep-2026): la corrida arrancó en la instancia
# vieja, murió con ella, y sin latido la pantalla esperó diez minutos.
LATIDO_ABANDONADO_S = 150.0
CONTRASTE_ABANDONADO_S = LATIDO_ABANDONADO_S
CONSULTA_ABANDONADA_S = LATIDO_ABANDONADO_S


def abandonada(doc: dict, ahora=time.time,
               abandonado: float = LATIDO_ABANDONADO_S) -> bool:
    """¿Una marca «en_curso» sin señales de vida?"""
    if not isinstance(doc, dict) or doc.get("estado") != "en_curso":
        return False
    try:
        ultimo = float(doc.get("latido") or doc.get("desde") or 0)
    except (TypeError, ValueError):
        ultimo = 0.0
    return ahora() - ultimo > abandonado


async def esperar_marca(huella: str, leer, que: str = "el contraste adelantado",
                        tope: float = 150.0, abandonado: float = CONTRASTE_ABANDONADO_S,
                        ahora=time.time, dormir=asyncio.sleep,
                        cada: float = 3.0) -> dict | None:
    """La marca que dejó una tarea del adelanto —{huella, estado, desde, …}—
    cuando está «listo» y es de ESTE adelanto; None si hay que hacer el
    trabajo aquí.

    `leer()` devuelve el dict guardado o None. Se espera sólo mientras la fila
    dice «en_curso» y el que lo calcula sigue vivo; nunca más de `tope`
    segundos, que es menos de lo que costaría calcularlo de nuevo.
    """
    t0 = ahora()
    avisado = False
    while True:
        doc = leer()
        if not isinstance(doc, dict) or doc.get("huella") != huella:
            return None
        estado = doc.get("estado")
        if estado == "listo":
            return doc
        if estado != "en_curso":
            return None
        if abandonada(doc, ahora, abandonado):
            print(f"   ⏳ {que} no dio señales en {abandonado:.0f} s: se hace aquí")
            return None
        if ahora() - t0 >= tope:
            print(f"   ⏳ {que} sigue en curso tras {tope:.0f} s de espera: se hace aquí")
            return None
        if not avisado:
            print(f"   ⏳ {que} sigue en curso: se espera")
            avisado = True
        await dormir(cada)


async def esperar_contraste(huella: str, leer, tope: float = 150.0,
                            ahora=time.time, dormir=asyncio.sleep,
                            cada: float = 3.0) -> list | None:
    """Los planteamientos contrastados que dejó el adelanto, o None si hay que
    calcularlos aquí."""
    doc = await esperar_marca(huella, leer, "el contraste adelantado", tope,
                              CONTRASTE_ABANDONADO_S, ahora, dormir, cada)
    if doc is None:
        return None
    items = doc.get("items")
    return list(items) if isinstance(items, list) else None


async def esperar_consulta(huella: str, leer, tope: float = 90.0,
                           ahora=time.time, dormir=asyncio.sleep,
                           cada: float = 2.0) -> bool:
    """¿La consulta automática del acervo de ESTE adelanto está lista? Espera
    mientras corre; False si hay que consultar aquí."""
    doc = await esperar_marca(huella, leer, "la consulta automática del acervo",
                              tope, CONSULTA_ABANDONADA_S, ahora, dormir, cada)
    return doc is not None
