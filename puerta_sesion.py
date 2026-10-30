"""QUIÉN PIDE LO DICE EL TOKEN, NO EL FORMULARIO.   (29-oct-2026)

EL INCIDENTE. Comprobado sobre la cuenta del dueño, sólo leyendo: sin ningún
token, la API de producción contestaba con los datos de una cuenta a quien
supiera su correo. /taller/estado daba su acceso y su cupo; /taller/en-curso,
sus asuntos con el nombre del quejoso; /taller/sise-pendiente, lo que esperaba
del SISE; y /taller/descargar entregaba el .docx entero del proyecto. Por la
misma vía se podía generar a costa de otro, borrarle un asunto
(/taller/olvidar) o usar el redactor, /chat-sentencia o la jurimetría con el
plan de otro. Y como `jdm.juridico@gmail.com` es de administración, decir ese
correo abría el taller sin tope y con las banderas de casa. Las rutas decidían
de quién eran los datos o la cuota con el `user_email` o el `user_id` que
viajaba en la petición, y ése lo escribe cualquiera.

UNA SOLA PUERTA, DE TODA LA APP. `puerta` es una dependencia global (main.py la
pone en `FastAPI(dependencies=...)`), así que una ruta nueva nace con ella; si
fuera endpoint por endpoint, una firma olvidada en ciento cuarenta sería el
mismo agujero. Las reglas:

  1. LO QUE DICE DE QUIÉN ES, SE DEMUESTRA. Si la petición trae `user_email` o
     `user_id` —en la URL, en la ruta, en el formulario o en el JSON— tiene que
     traer `Authorization: Bearer <access_token de Supabase>`, y el token tiene
     que ser de esa cuenta. Sin token, 401; con el de otra, 403. Se miran TODOS
     los valores y no el primero: si la URL dijera el correo propio y el
     formulario el ajeno, comprobar uno solo dejaría pasar al otro, que es el
     que lee el endpoint. Cada endpoint sigue leyendo `user_email` como
     siempre, pero ya demostrado.
  2. `EXIGEN_SESION`: token aunque la petición no diga de quién es. El taller
     entero (salvo su catálogo): un asunto se nombra por su número, y el número
     no es secreto.
  3. `NO_SE_MIRA`: el catálogo del taller, el aviso firmado de Apple, el
     renglón de registro de /chat/traba (sendBeacon no manda cabeceras) y la
     administración, donde el `user_id` de la ruta es A QUIÉN se administra y
     `_verify_admin` ya exige el token de un administrador.
  4. `TOLERAN_ANONIMO`: clientes ya publicados que dicen `user_id` SIN token —
     la app móvil en /analyze-document—. Ahí lo dicho sin token no se cree, pero
     tampoco se rechaza: se atiende como anónimo (`id_verificado` devuelve
     None). Romper la app no protege a nadie; atenderla como anónima, sí: sin
     token nadie gasta la cuota ni el plan de otro.

Lo que no dice de quién es y no está en `EXIGEN_SESION` pasa sin mirar el
token: las rutas públicas no gastan un viaje a Supabase.

CÓMO SE COMPRUEBA EL TOKEN. Se le pregunta a Supabase (`auth.get_user`), como
ya hacían la Bóveda, la cronología y /cuenta/eliminar: así un token de una
sesión cerrada también cae. Con caché por proceso de 60 s que nunca dura más
que el propio token, porque la pantalla del taller pregunta por /taller/plan
cada dos segundos. Antes de gastar el viaje se mira la forma: lo que no es un
JWT o ya venció se rechaza aquí. Supabase caído es 503, no 401: no es culpa de
quien pide y no se le dice que su sesión caducó.

LA PALANCA DE EMERGENCIA. `SESION_EXIGIR=0` deja pasar lo que llegue SIN token
(con aviso en el registro), por si aparece un cliente que no sabíamos que
existía. Un token de OTRA cuenta se rechaza siempre, con palanca o sin ella.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from fastapi import HTTPException, Request

# El cliente de administración de Supabase; main.py lo da con `configurar`,
# como el cliente o como una función que lo devuelve (así una prueba que
# sustituye `main.supabase_admin` sustituye también el de aquí).
_sb = None


def configurar(cliente) -> None:
    global _sb
    _sb = cliente


def _cliente():
    return _sb() if callable(_sb) and not hasattr(_sb, "auth") else _sb


EXIGIR = os.getenv("SESION_EXIGIR", "1") != "0"

# Los nombres con que una petición dice de quién es.
CAMPOS_CORREO = ("user_email",)
CAMPOS_ID = ("user_id",)

# Lo que llega en esos campos cuando el navegador no tenía a nadie: no es una
# cuenta, es la ausencia de una, y se trata como tal.
_VACIOS = frozenset({"", "null", "undefined", "none", "anonymous", "anon"})

# Rutas por su PLANTILLA (la de FastAPI, p. ej. "/quota/status/{user_id}").
EXIGEN_SESION_PREFIJOS = ("/taller/",)
EXIGEN_SESION = frozenset({
    # Escribe en el directorio de abogados; sin sesión cualquiera reescribía
    # el perfil de otro por su cédula, incluido «verificado».
    "/connect/lawyers/index",
})

NO_SE_MIRA = frozenset({
    # El catálogo del taller: no dice nada de nadie y la pantalla lo pide
    # antes de saber quién está delante. /taller/extension lo baja además el
    # servidor de la web (/api/complemento), sin sesión de nadie.
    "/taller/tipos",
    "/taller/reglas-surtimiento",
    "/taller/extension",
    # Administración: el user_id de la ruta es la cuenta administrada.
    "/admin/users/{user_id}/toggle-sentencia",
    "/admin/users/{user_id}/confirm-email",
    "/admin/users/{user_id}/block",
    "/admin/users/{user_id}/unblock",
    # Apple firma el aviso (apple_iap.py); no viene de ninguna sesión.
    "/api/apple/notificaciones",
    # Un renglón de registro por sendBeacon, que no puede mandar cabeceras;
    # no lee ni escribe nada de nadie (respuesta_guardada.py).
    "/chat/traba",
})

TOLERAN_ANONIMO = frozenset({
    # La app móvil publicada (iurexia-mobile, src/lib/api.ts) manda `user_id`
    # sin token. Sin token se atiende como anónima: límites del gratuito y sin
    # cobrarle a nadie. Cuando la app mande el token, sacar de aquí.
    "/analyze-document",
})

MENSAJE_SIN_SESION = (
    "Su sesión de Iurexia no llegó con la petición o ya caducó. Vuelva a cargar "
    "la página —su trabajo está guardado— y, si sigue, entre de nuevo en Iurexia.")
MENSAJE_AJENA = (
    "Esa petición va a nombre de otra cuenta. Vuelva a cargar la página con su "
    "sesión.")
MENSAJE_SIN_SUPABASE = (
    "No se puede comprobar la sesión ahora mismo. Inténtelo en un minuto.")

# Las del complemento del SISE: el que lo ve es el secretario en la pestaña
# del SISE, y lo que tiene que hacer es distinto.
_MENSAJES_RUTA = {
    "/taller/desde-sise": (
        "No se pudo saber de quién son estas constancias. Abra iurexia.com en "
        "una pestaña, entre con su cuenta y vuelva a pulsar: el complemento "
        "recoge la sesión solo. Si su complemento es anterior a la versión 3, "
        "descárguelo de nuevo desde el taller."),
}


@dataclass(frozen=True)
class Sesion:
    id: str
    correo: str


def enmascarar(correo: str) -> str:
    """Para el registro: lo justo para reconocer la cuenta, no el correo."""
    c = (correo or "").strip().lower()
    if "@" not in c:
        return "(sin correo)" if not c else c[:4] + "…"
    usuario, dominio = c.split("@", 1)
    return f"{usuario[:2]}…@{dominio}"


# ── El token ────────────────────────────────────────────────────────────────

_CACHE: Dict[str, Tuple[Optional[Sesion], float]] = {}
_TTL_BUENO = 60.0
_TTL_MALO = 30.0           # un token rechazado no vuelve a gastar viaje en 30 s
_TOPE_CACHE = 5000
_HOLGURA_EXP = 30.0        # relojes que no coinciden del todo con Supabase
_ESPERA_SUPABASE = 8.0


def _extraer(authorization: str) -> str:
    tok = (authorization or "").strip()
    if tok[:7].lower() == "bearer ":
        tok = tok[7:].strip()
    return tok


def _carga(tok: str) -> Optional[dict]:
    """La carga del JWT SIN verificarla: sólo sirve para descartar lo que no
    es un token y para que la caché no sobreviva al `exp`. Quien verifica es
    Supabase."""
    partes = tok.split(".")
    if len(partes) != 3 or not partes[1]:
        return None
    try:
        crudo = partes[1] + "=" * (-len(partes[1]) % 4)
        d = json.loads(base64.urlsafe_b64decode(crudo))
    except Exception:
        return None
    return d if isinstance(d, dict) else None


def _guardar(clave: str, sesion: Optional[Sesion], vence: float) -> None:
    if len(_CACHE) >= _TOPE_CACHE:
        ahora = time.monotonic()
        for k in [k for k, (_s, v) in _CACHE.items() if v <= ahora]:
            _CACHE.pop(k, None)
        if len(_CACHE) >= _TOPE_CACHE:
            _CACHE.clear()
    _CACHE[clave] = (sesion, vence)


def _no_vale(ruta: str = "") -> HTTPException:
    return HTTPException(401, _MENSAJES_RUTA.get(ruta, MENSAJE_SIN_SESION),
                         headers={"X-Sesion": "caducada"})


async def sesion_del_token(authorization: str, ruta: str = "") -> Optional[Sesion]:
    """La sesión que firma la petición; None si no trae token.

    401 si el token no vale (inventado, vencido, de una sesión cerrada). 503 si
    Supabase no contesta."""
    tok = _extraer(authorization)
    if not tok:
        return None
    clave = hashlib.sha256(tok.encode()).hexdigest()
    ahora = time.monotonic()
    guardado = _CACHE.get(clave)
    if guardado and guardado[1] > ahora:
        if guardado[0] is None:
            raise _no_vale(ruta)
        return guardado[0]

    carga = _carga(tok)
    if carga is None:
        raise _no_vale(ruta)
    exp = carga.get("exp")
    exp = float(exp) if isinstance(exp, (int, float)) else None
    if exp is not None and exp < time.time() - _HOLGURA_EXP:
        raise _no_vale(ruta)

    sb = _cliente()
    if sb is None:
        raise HTTPException(503, MENSAJE_SIN_SUPABASE)
    try:
        r = await asyncio.wait_for(asyncio.to_thread(sb.auth.get_user, tok),
                                   timeout=_ESPERA_SUPABASE)
    except asyncio.TimeoutError:
        print("   ‼️ SESION: Supabase tardó más de "
              f"{_ESPERA_SUPABASE:.0f} s en comprobar un token")
        raise HTTPException(503, MENSAJE_SIN_SUPABASE)
    except Exception as ex:
        estado = getattr(ex, "status", None)
        if isinstance(estado, int) and 400 <= estado < 500:
            _guardar(clave, None, ahora + _TTL_MALO)
            raise _no_vale(ruta)
        print(f"   ‼️ SESION: Supabase no contestó al comprobar un token "
              f"({type(ex).__name__})")
        raise HTTPException(503, MENSAJE_SIN_SUPABASE)

    u = getattr(r, "user", None)
    uid = str(getattr(u, "id", "") or "").strip().lower() if u else ""
    correo = str(getattr(u, "email", "") or "").strip().lower() if u else ""
    if not uid:
        _guardar(clave, None, ahora + _TTL_MALO)
        raise _no_vale(ruta)
    sesion = Sesion(id=uid, correo=correo)
    vence = ahora + _TTL_BUENO
    if exp is not None:
        vence = min(vence, ahora + max(0.0, exp - time.time()))
    _guardar(clave, sesion, vence)
    return sesion


# ── Lo que la petición dice ─────────────────────────────────────────────────

def _limpios(valores) -> List[str]:
    """Normalizados, sin los vacíos. Lo que no es texto —una lista, un objeto,
    un archivo con ese nombre de campo— NO se descarta: cuenta como un valor
    que no es de nadie, y así da 403 en vez de colarse sin mirar."""
    fuera: List[str] = []
    for v in valores:
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            fuera += _limpios(v)
            continue
        if not isinstance(v, (str, int, float)):
            v = f"<{type(v).__name__}>"
        s = str(v).strip().lower()
        if s not in _VACIOS:
            fuera.append(s)
    return fuera


async def dichos(request: Request) -> Tuple[List[str], List[str]]:
    """TODOS los correos y los ids que trae la petición: URL, ruta,
    formulario y JSON.

    El cuerpo ya lo leyó FastAPI antes de llegar aquí cuando el endpoint
    declara campos, y `form()` / `json()` devuelven lo guardado; cuando el
    endpoint lo lee a mano, lo que se lee aquí queda guardado para él."""
    correos: list = []
    ids: list = []
    q = request.query_params
    for c in CAMPOS_CORREO:
        correos += q.getlist(c)
    for c in CAMPOS_ID:
        ids += q.getlist(c)
    pp = request.path_params or {}
    correos += [pp[c] for c in CAMPOS_CORREO if c in pp]
    ids += [pp[c] for c in CAMPOS_ID if c in pp]

    # SE LEE COMO LO LEE QUIEN DESPUÉS LO USA. Si aquí se leyera menos que en
    # el endpoint, el correo que no se ve aquí pasaría sin demostrar: FastAPI
    # lee como JSON todo `application/json` y `application/*+json` aunque
    # traigan parámetros («; charset=utf-8»), Starlette lee formulario en
    # multipart y urlencoded, y /taller/origen lee JSON si el tipo CONTIENE
    # «application/json». Aquí se lee en todos esos casos.
    tipo = (request.headers.get("content-type") or "").lower()
    medio = tipo.split(";", 1)[0].strip()
    es_json = (medio == "application/json"
               or (medio.startswith("application/") and medio.endswith("+json"))
               or "application/json" in tipo)
    try:
        if medio in ("multipart/form-data", "application/x-www-form-urlencoded"):
            f = await request.form()
            for c in CAMPOS_CORREO:
                correos += f.getlist(c)
            for c in CAMPOS_ID:
                ids += f.getlist(c)
        elif es_json:
            cuerpo = await request.json()
            if isinstance(cuerpo, dict):
                correos += [cuerpo.get(c) for c in CAMPOS_CORREO if c in cuerpo]
                ids += [cuerpo.get(c) for c in CAMPOS_ID if c in cuerpo]
    except Exception:
        pass          # un cuerpo roto lo rechaza el propio endpoint con su 422
    return _limpios(correos), _limpios(ids)


def _plantilla(request: Request) -> str:
    ruta = request.scope.get("route")
    return getattr(ruta, "path", None) or request.url.path


def exige_sesion(plantilla: str) -> bool:
    if plantilla in NO_SE_MIRA:
        return False
    return plantilla in EXIGEN_SESION or plantilla.startswith(EXIGEN_SESION_PREFIJOS)


# ── La puerta ───────────────────────────────────────────────────────────────

async def puerta(request: Request) -> None:
    """Dependencia de toda la app. Ver el encabezado del módulo."""
    ruta = _plantilla(request)
    if ruta in NO_SE_MIRA:
        return
    exige = exige_sesion(ruta)
    correos, ids = await dichos(request)
    if not (exige or correos or ids):
        return

    sesion = await sesion_del_token(request.headers.get("authorization") or "", ruta)
    if sesion is None:
        if ruta in TOLERAN_ANONIMO and not exige:
            request.state.anonimo_tolerado = True
            print(f"   🔓 SESION: {request.method} {ruta} dice "
                  f"{'user_id' if ids else 'user_email'} sin token → se atiende "
                  f"como anónimo")
            return
        if not EXIGIR:
            print(f"   ⚠️ SESION: {request.method} {ruta} a nombre de "
                  f"{enmascarar(correos[0]) if correos else (ids[0][:8] + '…' if ids else 'nadie')} "
                  f"SIN token — pasa sólo porque SESION_EXIGIR=0")
            return
        raise HTTPException(401, _MENSAJES_RUTA.get(ruta, MENSAJE_SIN_SESION),
                            headers={"X-Sesion": "falta"})

    ajenos = ([c for c in correos if c != sesion.correo]
              + [i for i in ids if i != sesion.id])
    if ajenos:
        otro = ajenos[0]
        print(f"   ‼️ SESION: la sesión de {enmascarar(sesion.correo)} pidió "
              f"{request.method} {ruta} a nombre de "
              f"{enmascarar(otro) if '@' in otro else otro[:8] + '…'} — 403")
        raise HTTPException(403, MENSAJE_AJENA, headers={"X-Sesion": "ajena"})
    request.state.sesion = sesion


def sesion_de(request: Request) -> Optional[Sesion]:
    """La sesión que la puerta ya comprobó, si la hubo."""
    return getattr(request.state, "sesion", None)


def id_verificado(request, dicho: Optional[str]) -> Optional[str]:
    """El `user_id` con que debe atenderse una ruta de `TOLERAN_ANONIMO`: el
    dicho si la puerta lo comprobó contra el token; None si llegó sin token.

    Acepta cualquier objeto con `state` (o sin él): hay pruebas que llaman al
    endpoint como función con un pedido de mentira."""
    estado = getattr(request, "state", None)
    if estado is not None and getattr(estado, "anonimo_tolerado", False):
        return None
    return dicho
