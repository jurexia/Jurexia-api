"""
Verificación de compras hechas con Google Play Billing.

Es la pieza gemela de `apple_iap.py`, pero el mecanismo es distinto y conviene
saber por qué:

  · Apple **firma** el comprobante. Se puede verificar sin preguntarle nada a
    Apple: basta comprobar la firma contra su certificado raíz, que es público.
  · Google **no** firma nada equivalente. Devuelve un `purchaseToken` opaco que
    no dice nada por sí solo; hay que preguntarle a la Play Developer API si ese
    token corresponde a una suscripción viva. Es decir: aquí sí hacen falta
    credenciales de servidor.

De ahí que este módulo necesite una cuenta de servicio y el de Apple no.

─── Lo que hay que configurar una sola vez ──────────────────────────────────

1. Google Cloud → habilitar «Google Play Android Developer API».
2. Crear una cuenta de servicio y bajar su JSON.
3. Play Console → Usuarios y permisos → invitar al correo de esa cuenta de
   servicio con permiso «Ver datos financieros» y «Gestionar pedidos».
   (Este paso se olvida siempre y el síntoma es un 401 de la API sin más
   explicación.)
4. Poner el JSON completo en la variable de entorno
   `GOOGLE_PLAY_SERVICE_ACCOUNT` del servicio de Render.

Mientras el paso 4 no esté hecho, `verificar_compra` lanza `PlayNoConfigurado`
y el endpoint responde 503. **No** se concede el plan a ciegas: igual que con
Apple, la app no cierra la transacción si esto falla, así que Play la vuelve a
entregar y el reintento ocurre solo. Es preferible que una compra tarde en
activarse a regalar planes por una configuración a medias.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Optional

import httpx

PAQUETE = os.getenv("ANDROID_PACKAGE_NAME", "com.iurexia.app")
_ALCANCE = "https://www.googleapis.com/auth/androidpublisher"
_API = "https://androidpublisher.googleapis.com/androidpublisher/v3"


class PlayNoConfigurado(Exception):
    """Faltan las credenciales de la cuenta de servicio de Play."""


class CompraInvalida(Exception):
    """El token no corresponde a una suscripción válida de esta app."""


# ─── Productos ───────────────────────────────────────────────────────────────
#
# Espejo de PRODUCTOS_PLAY en la app (src/lib/storekit.ts). Si aquí y allá no
# coinciden, el abogado paga y no se le activa nada — mantenerlos a la par.
#
# Play no admite puntos en el id de una suscripción, por eso no son los mismos
# identificadores que en Apple. Platinum Anual sí está: el tope de precio que
# obligó a dejarlo fuera de la App Store es de Apple, no de Google.
PRODUCTOS: dict[str, str] = {
    "basico_mensual": "basico_monthly",
    "basico_anual": "basico_annual",
    "pro_mensual": "pro_monthly",
    "pro_anual": "pro_annual",
    "platinum_mensual": "platinum_monthly",
    "platinum_anual": "platinum_annual",
    "ultra_mensual": "ultra_secretarios",
}


@dataclass
class CompraVerificada:
    plan: str
    product_id: str
    transaction_id: str
    expira_ms: Optional[int]
    entorno: str
    #: El id de la cuenta de Iurexia que la app mandó al comprar, si vino.
    app_account_token: Optional[str]
    revocada: bool


# ─── Credenciales ────────────────────────────────────────────────────────────

_token_cache: dict[str, object] = {"valor": None, "expira": 0.0}


def _credenciales():
    crudo = os.getenv("GOOGLE_PLAY_SERVICE_ACCOUNT", "").strip()
    if not crudo:
        raise PlayNoConfigurado(
            "Falta GOOGLE_PLAY_SERVICE_ACCOUNT. Ver las instrucciones al inicio de google_play.py."
        )
    try:
        return json.loads(crudo)
    except json.JSONDecodeError as e:
        raise PlayNoConfigurado(f"GOOGLE_PLAY_SERVICE_ACCOUNT no es un JSON válido: {e}") from e


def _token_de_acceso() -> str:
    """
    Token OAuth de la cuenta de servicio, con caché.

    Se guarda hasta un minuto antes de su vencimiento: pedir uno nuevo en cada
    compra añadiría medio segundo a un flujo donde el usuario ya está esperando.
    """
    ahora = time.time()
    if _token_cache["valor"] and float(_token_cache["expira"]) > ahora:
        return str(_token_cache["valor"])

    datos = _credenciales()
    try:
        from google.oauth2 import service_account  # type: ignore
        from google.auth.transport.requests import Request  # type: ignore
    except ImportError as e:
        raise PlayNoConfigurado(
            "Falta la dependencia google-auth para hablar con la Play Developer API."
        ) from e

    cred = service_account.Credentials.from_service_account_info(datos, scopes=[_ALCANCE])
    cred.refresh(Request())
    _token_cache["valor"] = cred.token
    _token_cache["expira"] = (cred.expiry.timestamp() - 60) if cred.expiry else (ahora + 1800)
    return str(cred.token)


# ─── Verificación ────────────────────────────────────────────────────────────

def verificar_compra(purchase_token: str, product_id: Optional[str] = None) -> CompraVerificada:
    """
    Le pregunta a Google si este token es una suscripción viva de esta app.

    `product_id` es opcional: la API v2 devuelve la línea comprada, así que se
    puede deducir. Se acepta por si la app lo manda, para poder contrastarlo.
    """
    if not purchase_token:
        raise CompraInvalida("No llegó el comprobante de la compra.")

    token = _token_de_acceso()
    url = f"{_API}/applications/{PAQUETE}/purchases/subscriptionsv2/tokens/{purchase_token}"

    try:
        r = httpx.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=30.0)
    except httpx.HTTPError as e:
        # Fallo de red: que la app reintente, no que se rechace la compra.
        raise PlayNoConfigurado(f"No se pudo contactar a Google Play: {e}") from e

    if r.status_code == 401 or r.status_code == 403:
        raise PlayNoConfigurado(
            "Google rechazó las credenciales. Revisa que la cuenta de servicio esté "
            "invitada en Play Console con permiso para gestionar pedidos."
        )
    if r.status_code == 404:
        raise CompraInvalida("Google no reconoce este comprobante.")
    if r.status_code >= 400:
        raise CompraInvalida(f"Google respondió {r.status_code} al verificar la compra.")

    datos = r.json()

    estado = str(datos.get("subscriptionState", ""))
    # Sólo estos dos estados dan derecho al plan. Un `ON_HOLD` o un `PAUSED` no,
    # y un `CANCELED` con periodo vigente sí — por eso se mira también la fecha.
    vivos = {"SUBSCRIPTION_STATE_ACTIVE", "SUBSCRIPTION_STATE_IN_GRACE_PERIOD"}

    lineas = datos.get("lineItems") or []
    if not lineas:
        raise CompraInvalida("El comprobante no trae ninguna suscripción.")
    linea = lineas[0]
    sku = str(linea.get("productId") or product_id or "")

    plan = PRODUCTOS.get(sku)
    if not plan:
        raise CompraInvalida(f"El producto «{sku}» no es de Iurexia.")

    expira_ms: Optional[int] = None
    caduca = linea.get("expiryTime")
    if caduca:
        # Viene como RFC 3339; se pasa a milisegundos para igualar a Apple.
        from datetime import datetime
        try:
            expira_ms = int(
                datetime.fromisoformat(str(caduca).replace("Z", "+00:00")).timestamp() * 1000
            )
        except ValueError:
            expira_ms = None

    vencida = bool(expira_ms and expira_ms < time.time() * 1000)
    revocada = estado not in vivos or vencida

    return CompraVerificada(
        plan=plan,
        product_id=sku,
        transaction_id=str(datos.get("latestOrderId") or purchase_token[:32]),
        expira_ms=expira_ms,
        entorno="produccion" if not datos.get("testPurchase") else "sandbox",
        # Es el `obfuscatedAccountId` que la app mandó al comprar.
        app_account_token=(datos.get("externalAccountIdentifiers") or {}).get(
            "obfuscatedExternalAccountId"
        ),
        revocada=revocada,
    )
