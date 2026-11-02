"""EL ORDEN DE LOS PLANES Y LO QUE UNA TIENDA PUEDE ESCRIBIR.   (2-nov-2026)

`PLAN_RANK` es el espejo de `PLAN_RANK` en src/lib/stripe.ts del front: si
cambia allá, cambia aquí. Estudio va en 8, sobre Ultra (7), por precio.

LA TIENDA NO PISA UN ESTUDIO PAGADO EN LA WEB (A8 del inventario). Las
compras de App Store y Google Play escriben `subscription_type` directo
(`_escribir_plan` en main.py), siempre. Con Estudio eso rompe dos cosas:
  · quien paga Estudio en la web y compra «Pro» en la app instalada —que no
    conoce Estudio, lo trata como gratuito y le ofrece comprar— quedaba
    escrito como `pro_monthly` y el trigger le dejaba 140 consultas;
  · cuando caduca una suscripción vieja de la tienda (EXPIRED, REFUND…) se
    escribía `gratuito` encima del Estudio que paga en Stripe.

La regla: si el perfil es Estudio y tiene una suscripción de Stripe viva
(`stripe_subscription_id`), la tienda sólo escribe un plan ESTRICTAMENTE
mayor (hoy no hay ninguno: Estudio es el 8). Sin suscripción de Stripe, la
tienda manda como siempre (ella dio el plan y ella lo quita).
`stripe_subscription_id` sí dice «viva»: el webhook la borra al cancelar.

SÓLO ESTUDIO, Y A PROPÓSITO (revisión de la API del 2-nov, hallazgo 3). La
primera versión aplicaba la regla a todo perfil con Stripe: ese día eran 290
de Básico, Pro, Platinum y Ultra, y con ella una compra de igual o menor
rango y los avisos de caducidad de la tienda dejaban de escribirles el plan.
La decisión del 2-nov es que todo lo que existe se queda exactamente como
está, y ese cambio no estaba aprobado. Para los demás planes la tienda
escribe como en main. Si algún día se quiere para todos, va aparte y con
el visto bueno de David.

Lo que esta regla NO resuelve: el perfil no guarda de dónde vino cada plan.
La línea «🍎 Tienda → … NO escribe» de `_escribir_plan` es la que se revisa
a mano.
"""
from __future__ import annotations

from typing import Optional

from freno_estudio import es_estudio

PLAN_RANK = {
    "gratuito": 0,
    "basico_monthly": 1,
    "basico_annual": 2,
    "pro_monthly": 3,
    "pro_annual": 4,
    "platinum_monthly": 5,
    "platinum_annual": 6,
    "ultra_secretarios": 7,
    "platinum_estudio": 8,
}


def rango(plan: Optional[str]) -> int:
    """Un plan desconocido cuenta como gratuito."""
    return PLAN_RANK.get((plan or "").strip().lower(), 0)


def tienda_puede_escribir(actual: Optional[str], stripe_subscription_id: Optional[str],
                          nuevo: Optional[str]) -> bool:
    """¿Puede una compra o un aviso de la tienda dejar `nuevo` en el perfil?
    Sí, siempre, salvo encima de un Estudio con Stripe vivo."""
    if not es_estudio(actual) or not (stripe_subscription_id or "").strip():
        return True
    return rango(nuevo) > rango(actual)
