"""La sesión de una cuenta de casa, para los bancos que llaman al taller.

Desde el 29-oct-2026 /taller/* exige `Authorization: Bearer <token de
Supabase>` y que su correo sea el `user_email` (ver puerta_sesion.py). Los
bancos —banco_kingston.py, banco_estudio.py— hablaban con producción sólo con
el correo, y ese correo es justo lo que ya no basta. (Escrito el 29-sep para
una puerta que no llegó a desplegarse; se trae tal cual.)

CÓMO SE SACA. Con la llave de servicio se pide un enlace mágico de
administración y su testigo se canjea AQUÍ contra /auth/v1/verify: la web
entra con PKCE, así que el enlace no se puede abrir en un navegador. Es la
misma receta de video-nodos/grabar45.py. No sale ningún correo.

SÓLO CUENTAS DE CASA. Sacar la sesión de cualquier cuenta con la llave de
servicio sería suplantarla; aquí se niega todo lo que no sea @iurexia.com.

EL TOKEN DURA UNA HORA y una corrida del banco dura más: por eso se pide la
cabecera EN CADA PETICIÓN y no al crear el cliente. Se renueva diez minutos
antes de caducar. Nunca se imprime.
"""
import json
import os
import threading
import time
import urllib.request

_DOMINIO_CASA = "@iurexia.com"
_MARGEN_S = 600
_guardadas: dict = {}
_cerrojo = threading.Lock()


class CuentaNoDeCasa(ValueError):
    pass


def _llaves():
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
    except Exception:
        pass
    url = (os.getenv("SUPABASE_URL") or "").rstrip("/")
    servicio = os.getenv("SUPABASE_SERVICE_KEY") or ""
    anonima = (os.getenv("SUPABASE_ANON_KEY")
               or os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY") or servicio)
    if not (url and servicio):
        raise RuntimeError("Faltan SUPABASE_URL / SUPABASE_SERVICE_KEY: sin ellas no "
                           "hay sesión y el taller contesta 401.")
    return url, servicio, anonima


def _canjear(correo: str) -> tuple:
    url, servicio, anonima = _llaves()
    req = urllib.request.Request(
        f"{url}/auth/v1/admin/generate_link",
        data=json.dumps({"type": "magiclink", "email": correo}).encode(),
        headers={"apikey": servicio, "Authorization": f"Bearer {servicio}",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.load(r)
    testigo = d.get("hashed_token") or (d.get("properties") or {}).get("hashed_token")
    if not testigo:
        raise RuntimeError(f"El enlace de {correo} no trajo testigo.")
    req = urllib.request.Request(
        f"{url}/auth/v1/verify",
        data=json.dumps({"type": "magiclink", "token_hash": testigo}).encode(),
        headers={"apikey": anonima, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        s = json.load(r)
    tok = s.get("access_token")
    if not tok:
        raise RuntimeError(f"El canje de {correo} no devolvió sesión.")
    vence = float(s.get("expires_at") or (time.time() + float(s.get("expires_in") or 3600)))
    return tok, vence


def cabecera_sesion(correo: str) -> dict:
    """{"Authorization": "Bearer …"} para `correo`, renovada si le queda poco."""
    c = (correo or "").strip().lower()
    if not c.endswith(_DOMINIO_CASA):
        raise CuentaNoDeCasa(f"«{correo}» no es una cuenta de casa: los bancos sólo "
                             f"sacan sesión de cuentas {_DOMINIO_CASA}.")
    with _cerrojo:
        tok, vence = _guardadas.get(c, (None, 0.0))
        if not tok or time.time() > vence - _MARGEN_S:
            tok, vence = _canjear(c)
            _guardadas[c] = (tok, vence)
    return {"Authorization": f"Bearer {tok}"}
