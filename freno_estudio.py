"""EL PLAN ESTUDIO Y SU FRENO DE 560.   (2-nov-2026)

Estudio (`platinum_estudio`, 1,499 al mes) es Platinum sin cupo mensual de
consultas. Lo que lo hace posible es este freno (ESTUDIO-PRECIOS.md §4):

  · las primeras 560 consultas de cada periodo funcionan EXACTAMENTE como en
    Platinum: motores de élite, documentos escaneados (OCR), Bóveda y
    Cronología. Lo que pasa por OCR cuenta dentro de esas 560;
  · desde la 561 —`queries_used` ≥ 560 antes de cobrar la consulta— el chat,
    los documentos y la redacción siguen sin límite con los motores de Pro
    (`pro_monthly`), y la Bóveda, la Cronología y el OCR esperan al siguiente
    periodo.

Por qué: sin el freno, cada Estudio usaría el motor de élite hasta el tope y
el escenario central pasaba de +7.1% a −11.3%. El chat cuesta ~0.32 por
consulta; lo caro es el motor de documentos de élite, la redacción de élite
(6.55) y el OCR (16.2 un documento de 600 hojas sin caché). Con el freno,
Estudio nunca da menos que Platinum y su exposición es la misma.

EL TOPE NO ES −1. `queries_limit` de Estudio es 3,100 (100 al día × 31): con
−1 la web y la app instalada calculan «te quedan 0» y bloquean el chat. La
pantalla dice «Ilimitadas». Ese número vive en la base (trigger) y en el
PLAN_CONFIG del front; la API no lo lee.

EL FRENO SÓLO MIRA EL PLAN Y EL CONTADOR. Ni la administración ni la cortesía
(testers, Jenica) lo saltan: una cuenta de prueba con Estudio y 560 usadas
tiene que frenar igual que la de un cliente, o el freno no se puede probar en
producción. Donde la administración no paga (Bóveda, /analyze-document) su
contador no crece y el freno no le llega.

SI EL CONTADOR NO SE PUEDE LEER, SE ATIENDE COMO PLATINUM Y SE GRITA. Es la
decisión del 2-nov: un tropiezo de la base no deja al abogado sin su plan.
La línea «🚨 FRENO DE ESTUDIO SIN CONTADOR» es la que hay que buscar en los
registros de Render para contarlo.

DÓNDE SE APLICA (precios-nov2/estudio-hoy/api/decisiones.md):
  · /chat: el escalón Platinum de la redacción cae al de Pro y la auditoría
    de sentencias se redacta con el motor de Pro. Decide con lo que contestó
    `consume_query` al cobrar esa misma consulta (`frenado_tras_cobro`): ni
    otro viaje a la base ni carrera con el cobro, que corre en paralelo.
  · /analyze-document: motor de documentos de Pro y sin OCR pagado (la caché
    del OCR sí). Lee `queries_used` en la misma lectura del perfil.
  · Jurimetría con adjuntos: sin OCR pagado.
  · Bóveda y Cronología: sólo lo que quepa en lo que queda de las 560
    (`exigir_elite`, desde `cobro_consultas.cobrar_consultas(plan=…)`).
"""
from __future__ import annotations

from typing import Optional

from fastapi import HTTPException

PLAN_ESTUDIO = "platinum_estudio"

# Las consultas de cada periodo con la IA Jurídica de Élite. La tarjeta de
# /precios y la línea de uso razonable dicen el mismo número: si cambia aquí,
# cambia allá.
CONSULTAS_ELITE = 560

# Los flujos al mes que promete la tarjeta (Platinum: 60). flujo_agente.py lo
# lee de aquí; la base (`public.limite_flujos`, que es la que cobra) y el front
# (`limiteFlujos`) tienen que decir lo mismo, o la tarjeta promete lo que no
# se entrega.
FLUJOS_AL_MES = 100

# El aviso, con las palabras aprobadas el 2-nov-2026. Sin fechas: el fin del
# periodo (`subscription_end`) no lo mantiene nadie —medido ese día: 5 de 316
# perfiles de pago lo tienen—, así que una fecha aquí sería inventada.
MENSAJE_AGOTADAS = (
    f"Sus {CONSULTAS_ELITE} consultas con la IA Jurídica de Élite de este mes ya se usaron; "
    "Bóveda, Cronología y los documentos escaneados vuelven con su siguiente periodo. "
    "Las consultas y la redacción siguen sin límite.")


def es_estudio(plan: Optional[str]) -> bool:
    return (plan or "").strip().lower() == PLAN_ESTUDIO


def _contador(n) -> Optional[int]:
    """`queries_used` como entero, o None si no se pudo leer (nulo, texto
    raro). Cero es un contador leído, no uno ausente."""
    if n is None or isinstance(n, bool):
        return None
    try:
        return max(0, int(n))
    except (TypeError, ValueError):
        return None


def sin_contador(usadas, donde: str) -> None:
    print(f"   🚨 FRENO DE ESTUDIO SIN CONTADOR{f' ({donde})' if donde else ''}: "
          f"queries_used ilegible ({usadas!r}) — se atiende como Platinum")


def frenado(plan: Optional[str], usadas_previas, donde: str = "") -> bool:
    """¿Esta consulta ya va con el freno puesto? `usadas_previas` es
    `queries_used` ANTES de cobrar la consulta de ahora: con 560 ya gastadas,
    la siguiente es la 561. Cualquier otro plan, nunca. Sin contador legible,
    no frena y queda en el registro."""
    if not es_estudio(plan):
        return False
    n = _contador(usadas_previas)
    if n is None:
        sin_contador(usadas_previas, donde)
        return False
    return n >= CONSULTAS_ELITE


def elite_restantes(plan: Optional[str], usadas) -> Optional[int]:
    """Lo que le queda de las 560 en este periodo. None si el plan no tiene
    freno (todos menos Estudio) o si el contador no se pudo leer: en los dos
    casos, quien llama sigue como con Platinum."""
    if not es_estudio(plan):
        return None
    n = _contador(usadas)
    return None if n is None else max(0, CONSULTAS_ELITE - n)


def usadas_antes_del_cobro(cobro: Optional[dict]) -> Optional[int]:
    """`queries_used` antes de ESTA consulta, sacado de lo que devolvió
    `consume_query` (Supabase). Esa función contesta `used` ya sumada la
    consulta si la cobró del plan (`used` = previas + 1), y sin sumarla si la
    cobró de una recarga (`de_recarga`, el plan ya estaba en su tope). None si
    no hubo cobro que leer: el chat sigue aunque `consume_query` falle."""
    if not isinstance(cobro, dict) or not cobro.get("allowed"):
        return None
    usadas = _contador(cobro.get("used"))
    if usadas is None:
        return None
    return usadas if cobro.get("de_recarga") else max(0, usadas - 1)


def frenado_tras_cobro(cobro: Optional[dict]) -> Optional[bool]:
    """El freno de /chat, con el plan y el contador que `consume_query` leyó
    bajo cerrojo para cobrar esta misma consulta.

    None = no hubo cobro que leer (la base tropezó, o no hay usuario): quien
    llama decide con la lectura del perfil que ya tenga a mano, y si no tiene
    ninguna sigue como Platinum. Es el criterio del chat, que no deja al
    abogado sin respuesta por un tropiezo de la base."""
    usadas = usadas_antes_del_cobro(cobro)
    if usadas is None:
        return None
    return es_estudio(cobro.get("subscription_type")) and usadas >= CONSULTAS_ELITE


def mensaje_insuficientes(restantes: int, necesarias: int) -> str:
    """El aviso cuando quedan algunas de las 560 pero no las que pide la
    tarea (una Cronología de 300 hojas son 30); sin ninguna, el aprobado."""
    restantes = _contador(restantes) or 0
    necesarias = max(1, _contador(necesarias) or 0)
    if restantes <= 0:
        return MENSAJE_AGOTADAS
    n = f"{necesarias} consulta" + ("" if necesarias == 1 else "s")
    q = f"le queda {restantes}" if restantes == 1 else f"le quedan {restantes}"
    return (f"Esta tarea necesita {n} con la IA Jurídica de Élite y de sus "
            f"{CONSULTAS_ELITE} de este mes {q}. Puede hacerla con menos páginas, o "
            f"completa con su siguiente periodo. Las consultas y la redacción siguen sin límite.")


class SinElite(HTTPException):
    """El 402 del freno con el detalle en texto (así lo pintan
    /analyze-document y Jurimetría). Es su propia clase para que quien lea un
    documento pueda dejarlo pasar intacto entre los `except Exception` que
    convierten cualquier fallo del OCR en un 500."""

    def __init__(self, mensaje: str = MENSAJE_AGOTADAS):
        super().__init__(status_code=402, detail=mensaje)


def rechazo(restantes: int = 0, necesarias: int = 1) -> HTTPException:
    """El 402 del freno con el dict de siempre del cobro (`codigo`, `mensaje`,
    …), como `saldo_insuficiente`: la Bóveda y la Cronología de la web pintan
    `detail.mensaje` (src/lib/boveda.ts y cronologia.ts). Con su propio
    `codigo` para que la pantalla no ofrezca una recarga: a Estudio no le abre
    ni la Bóveda ni la Cronología."""
    mensaje = mensaje_insuficientes(restantes, necesarias)
    return HTTPException(402, {"codigo": "freno_estudio", "mensaje": mensaje,
                               "necesarias": max(1, _contador(necesarias) or 0),
                               "disponibles": _contador(restantes) or 0,
                               "consultas_elite": CONSULTAS_ELITE})


def exigir_elite(plan: Optional[str], usadas, necesarias: int = 1, donde: str = "") -> None:
    """402 `freno_estudio` si es Estudio y la tarea no cabe en lo que le queda
    de las 560. Los demás planes pasan sin más: su tope lo cuida el cobro.
    Sin contador legible pasa también (como Platinum) y queda en el registro."""
    if not es_estudio(plan):
        return
    restantes = elite_restantes(plan, usadas)
    if restantes is None:
        sin_contador(usadas, donde)
        return
    if restantes < max(1, _contador(necesarias) or 0):
        raise rechazo(restantes, necesarias)


def resumen(plan: Optional[str], usadas) -> Optional[dict]:
    """Para la pantalla (Bóveda y Cronología): cuánto queda de las 560 y, si
    ya no queda nada, el aviso. None si el plan no tiene freno o si el
    contador no se pudo leer."""
    restantes = elite_restantes(plan, usadas)
    if restantes is None:
        return None
    return {"consultas_elite": CONSULTAS_ELITE, "restantes": restantes,
            "mensaje": MENSAJE_AGOTADAS if restantes <= 0 else None}
