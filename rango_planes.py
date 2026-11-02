"""EL ORDEN DE LOS PLANES Y LO QUE UNA TIENDA PUEDE ESCRIBIR.   (2-nov-2026)

`PLAN_RANK` es el espejo de `PLAN_RANK` en src/lib/stripe.ts del front: si
cambia allá, cambia aquí. Estudio va en 8, sobre Ultra (7), por precio.

LA TIENDA NO PISA UN PLAN DE LA WEB MAYOR (A8 del inventario). Las compras de
App Store y Google Play escriben `subscription_type` directo
(`_escribir_plan` en main.py). Hasta hoy escribían siempre, y con Estudio eso
rompe dos cosas:
  · quien paga Estudio en la web y compra «Pro» en la app quedaba escrito
    como `pro_monthly` y el trigger le dejaba 140 consultas;
  · cuando caduca una suscripción vieja de la tienda (EXPIRED, REFUND…) se
    escribía `gratuito` encima del plan que paga en Stripe.
La app instalada (1.5.0) no conoce Estudio: su pantalla lo trata como
gratuito y le ofrece comprar.

La regla: si el perfil tiene una suscripción de Stripe viva
(`stripe_subscription_id`), la tienda sólo escribe un plan ESTRICTAMENTE
mayor. Igual rango es el mismo plan: ya está escrito y no hay nada que hacer.
Sin suscripción de Stripe, la tienda manda como siempre (ella dio el plan y
ella lo quita).

`stripe_subscription_id` sí dice «viva»: el webhook la borra al cancelar.
Medido el 2-nov-2026: 290 perfiles la tienen y ninguno está en `gratuito`.

Lo que esta regla NO resuelve: quien paga Pro en Stripe y Platinum en la
tienda conserva Platinum cuando la tienda avisa que caducó, porque el perfil
no guarda de dónde vino cada plan. La línea «🍎 Tienda → … NO escribe» de
`_escribir_plan` es la que se revisa a mano.
"""
from __future__ import annotations

from typing import Optional

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
    """¿Puede una compra o un aviso de la tienda dejar `nuevo` en el perfil?"""
    if not (stripe_subscription_id or "").strip():
        return True
    return rango(nuevo) > rango(actual)
