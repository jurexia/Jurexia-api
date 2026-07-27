"""
apple_iap.py — Suscripciones compradas dentro de la app de iPhone (StoreKit 2).

Por qué existe: la directriz 3.1.1 de Apple obliga a vender con compras dentro
de la app el contenido digital que se consume en ella. La app móvil cobra con
StoreKit y manda aquí el comprobante; este módulo lo verifica y escribe el plan
en `user_profiles.subscription_type`, exactamente el mismo campo que ya actualiza
el webhook de Stripe. Así un usuario que compró en el iPhone ve su plan al abrir
iurexia.com, y al revés.

─── Cómo se verifica (y por qué NO hace falta la llave .p8) ──────────────────

Apple firma cada transacción con un JWS cuyo encabezado trae la cadena de
certificados (`x5c`) que sube hasta "Apple Root CA - G3". Verificar esa firma
contra el certificado raíz —que es público y va versionado en `certs/apple/`—
demuestra criptográficamente que el comprobante lo emitió Apple y que nadie lo
alteró. Eso basta para conceder el plan.

La llave `.p8` de la App Store Server API sólo sería necesaria para *consultarle*
cosas a Apple (p. ej. "¿sigue activa esta suscripción?"). Como Apple nos avisa
de cada cambio por App Store Server Notifications, no se consulta nada y no hace
falta esa llave. Un secreto menos que custodiar.

─── La regla de oro ─────────────────────────────────────────────────────────

El cliente **nunca** decide qué plan tiene. La app manda el comprobante, este
módulo decide, y sólo si aquí sale bien la app cierra la transacción con Apple.
Si algo falla, la transacción queda abierta y StoreKit la vuelve a entregar en el
siguiente arranque: el reintento es automático y nadie se queda pagando sin plan.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from appstoreserverlibrary.models.Environment import Environment
from appstoreserverlibrary.models.JWSTransactionDecodedPayload import (
    JWSTransactionDecodedPayload,
)
from appstoreserverlibrary.models.ResponseBodyV2DecodedPayload import (
    ResponseBodyV2DecodedPayload,
)
from appstoreserverlibrary.signed_data_verifier import SignedDataVerifier, VerificationException

BUNDLE_ID = os.getenv("APPLE_BUNDLE_ID", "com.iurexia.app")
# El "Apple ID" numérico de la app en App Store Connect. La librería lo exige
# para verificar en Producción (evita que un comprobante de otra app pase).
APP_APPLE_ID = int(os.getenv("APPLE_APP_ID", "6794576234"))

_CERT_DIR = Path(__file__).parent / "certs" / "apple"


class CompraInvalida(Exception):
    """El comprobante no es de fiar: firma mala, otra app, o producto ajeno."""


# ─── Productos ───────────────────────────────────────────────────────────────

# Espejo de PRODUCTOS_APPLE en la app móvil (src/lib/storekit.ts). Si aquí y allá
# no coinciden, el usuario paga y no se le activa nada — mantenerlos a la par.
#
# Ojo: `platinum_annual` NO está, y es a propósito. La rejilla de precios de
# Apple en México tope a $1,999 MXN para esa suscripción, muy por debajo de lo
# que Iurexia necesita cobrar ($5,990/año), así que ese plan sólo se vende por
# Stripe. Si algún día Apple sube el tope, se da de alta el precio allá y se
# agrega la línea aquí y en la app.
PLAN_POR_PRODUCTO: dict[str, str] = {
    "com.iurexia.app.basico.mensual": "basico_monthly",
    "com.iurexia.app.basico.anual": "basico_annual",
    "com.iurexia.app.pro.mensual": "pro_monthly",
    "com.iurexia.app.pro.anual": "pro_annual",
    "com.iurexia.app.platinum.mensual": "platinum_monthly",
    "com.iurexia.app.ultra.mensual": "ultra_secretarios",
}


# ─── Verificadores ───────────────────────────────────────────────────────────


def _raices() -> list[bytes]:
    """Los certificados raíz de Apple, leídos del repo (son públicos)."""
    if not _CERT_DIR.is_dir():
        raise RuntimeError(f"Falta el directorio de certificados de Apple: {_CERT_DIR}")
    raices = [p.read_bytes() for p in sorted(_CERT_DIR.glob("*.cer"))]
    if not raices:
        raise RuntimeError(f"No hay certificados .cer en {_CERT_DIR}")
    return raices


# Se construyen una sola vez: leer y parsear los certificados en cada petición
# sería tirar CPU a la basura. Son objetos sin estado, seguros de compartir.
_verificadores: dict[Environment, SignedDataVerifier] = {}


def _verificador(entorno: Environment) -> SignedDataVerifier:
    if entorno not in _verificadores:
        _verificadores[entorno] = SignedDataVerifier(
            root_certificates=_raices(),
            # Las comprobaciones en línea (OCSP) añaden latencia y un punto de
            # falla externo en el camino crítico del cobro. La cadena firmada ya
            # prueba el origen; se dejan apagadas a propósito.
            enable_online_checks=False,
            environment=entorno,
            bundle_id=BUNDLE_ID,
            app_apple_id=APP_APPLE_ID if entorno == Environment.PRODUCTION else None,
        )
    return _verificadores[entorno]


def _verificar_en_ambos_entornos(verificar):
    """
    Intenta Producción y, si no, Sandbox.

    Hace falta porque las compras de TestFlight y las del revisor de Apple son
    de Sandbox, mientras que las de usuarios reales son de Producción, y el
    verificador rechaza el comprobante si el entorno no coincide. Probar los dos
    es la forma recomendada de que el mismo backend sirva para ambos.
    """
    ultimo_error: Optional[Exception] = None
    for entorno in (Environment.PRODUCTION, Environment.SANDBOX):
        try:
            return verificar(_verificador(entorno))
        except VerificationException as err:
            ultimo_error = err
    raise CompraInvalida(
        f"Apple no reconoce el comprobante (ni en producción ni en sandbox): {ultimo_error}"
    )


# ─── Resultado ───────────────────────────────────────────────────────────────


@dataclass
class CompraVerificada:
    plan: str
    product_id: str
    transaction_id: str
    original_transaction_id: str
    expira_ms: Optional[int]
    entorno: str
    #: UUID de la cuenta de Iurexia que la app mandó al comprar, si vino.
    app_account_token: Optional[str]
    revocada: bool


def _a_compra(tx: JWSTransactionDecodedPayload) -> CompraVerificada:
    if tx.bundleId != BUNDLE_ID:
        raise CompraInvalida(f"El comprobante es de otra app ({tx.bundleId}).")

    plan = PLAN_POR_PRODUCTO.get(tx.productId or "")
    if not plan:
        raise CompraInvalida(f"Producto desconocido: {tx.productId}")

    return CompraVerificada(
        plan=plan,
        product_id=tx.productId,
        transaction_id=str(tx.transactionId),
        original_transaction_id=str(tx.originalTransactionId),
        expira_ms=tx.expiresDate,
        entorno=str(tx.rawEnvironment or ""),
        app_account_token=str(tx.appAccountToken) if tx.appAccountToken else None,
        # `revocationDate` se llena cuando Apple reembolsó o retiró la compra.
        revocada=tx.revocationDate is not None,
    )


def verificar_transaccion(jws: str) -> CompraVerificada:
    """
    Comprueba el JWS de una compra y devuelve a qué plan da derecho.

    Lanza `CompraInvalida` si la firma no es de Apple, si el comprobante es de
    otra app o si el producto no es uno de los nuestros.
    """
    if not jws or not isinstance(jws, str):
        raise CompraInvalida("Falta el comprobante de compra.")

    tx = _verificar_en_ambos_entornos(
        lambda v: v.verify_and_decode_signed_transaction(jws)
    )
    return _a_compra(tx)


def verificar_notificacion(signed_payload: str) -> ResponseBodyV2DecodedPayload:
    """Comprueba una App Store Server Notification V2 y la devuelve decodificada."""
    if not signed_payload or not isinstance(signed_payload, str):
        raise CompraInvalida("Falta el cuerpo firmado de la notificación.")

    return _verificar_en_ambos_entornos(
        lambda v: v.verify_and_decode_notification(signed_payload)
    )


# ─── Qué hacer con cada aviso de Apple ───────────────────────────────────────

#: Avisos tras los cuales el usuario **tiene** derecho al plan del comprobante.
NOTIFICACIONES_QUE_DAN_PLAN = {
    "SUBSCRIBED",          # se suscribió (o resuscribió)
    "DID_RENEW",           # se renovó y cobró
    "OFFER_REDEEMED",      # canjeó una oferta
    "DID_CHANGE_RENEWAL_PREF",  # cambió de plan; el comprobante trae el nuevo
    "RENEWAL_EXTENDED",    # Apple le extendió el periodo
    "PRICE_INCREASE",      # sólo si aceptó; el plan sigue vivo
    "MIGRATION",
    # Apple echó atrás un reembolso: el usuario se queda con lo que compró, así
    # que hay que devolverle el plan que le quitamos al llegar el REFUND.
    "REFUND_REVERSED",
}

#: Avisos tras los cuales el usuario **pierde** el plan y vuelve a gratuito.
NOTIFICACIONES_QUE_QUITAN_PLAN = {
    "EXPIRED",             # se acabó y no renovó
    "GRACE_PERIOD_EXPIRED",  # se acabó el periodo de gracia sin cobrar
    "REFUND",              # Apple devolvió el dinero
    "REVOKE",              # se retiró el acceso (p. ej. compartir en familia)
}

#: Avisos que NO cambian el acceso: sólo informan.
#: `DID_CHANGE_RENEWAL_STATUS` es el caso importante — que alguien apague la
#: renovación automática **no** le quita el plan hoy; lo conserva hasta que
#: expire, y entonces llega `EXPIRED`. Cortarle aquí sería quitarle algo que ya
#: pagó.
NOTIFICACIONES_INFORMATIVAS = {
    "DID_CHANGE_RENEWAL_STATUS",
    "DID_FAIL_TO_RENEW",   # Apple sigue reintentando; aún no se pierde nada
    "REFUND_DECLINED",
    "CONSUMPTION_REQUEST",
    "TEST",
    "METADATA_UPDATE",
    "PRICE_CHANGE",
}
