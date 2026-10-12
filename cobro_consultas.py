"""COBRO PROPORCIONAL DE CONSULTAS Y PUERTA PLATINUM.   (oct-2026)

David: «esto consumirá las consultas en proporción, sino el usuario con
expedientes completos rebasaría el costo en servicios que lo que paga por su
plan, desde luego que esto es exclusivo de platinum».

Lo comparten la Bóveda y la Cronología del expediente: una sola puerta y un
solo cobro, del MISMO contador que el chat (`queries_used`, y cuando se acaba,
`consultas_recargadas`). Las funciones de Postgres están en
migration_cobro_consultas.sql:

  consultas_disponibles(user)          → cuánto puede gastar ahora
  consumir_consultas(user, n, motivo)  → todo o nada; deja un movimiento
  devolver_consultas(movimiento, n?)   → regresa a su origen, nunca de más

Los administradores no pagan (igual que en el chat): a ellos no se les cobra.

Este módulo no importa main.py: crea su propio cliente de Supabase con la
clave de servicio la primera vez que se le pide, o usa el que le pasen con
`configurar()`. Así se prueba sin el arranque de la API.
"""
from __future__ import annotations

import asyncio
import math
import os
import threading
from dataclasses import dataclass
from typing import Any, Optional

from fastapi import HTTPException

try:  # la misma lista que usa el chat para «Redacción Platinum»
    from esfuerzo_redaccion import PLANES_PLATINUM
except Exception:  # pragma: no cover
    PLANES_PLATINUM = {"platinum_monthly", "platinum_annual", "ultra_secretarios"}

# Igual que ADMIN_EMAILS de main.py: la base va fija en el código.
_ADMIN_BASE = {"administracion@iurexia.com", "jdm.juridico@gmail.com"}

_sb = None
_admins: Optional[set] = None


def configurar(supabase_admin=None, admin_emails: Optional[set] = None) -> None:
    """main.py le pasa su cliente y su lista de administradores al arrancar."""
    global _sb, _admins  # noqa: PLW0603
    if supabase_admin is not None:
        _sb = supabase_admin
    if admin_emails is not None:
        _admins = {e.strip().lower() for e in admin_emails if e}


_por_hilo = threading.local()


def _cliente():
    """El cliente de main.py si lo prestó (ya es uno por hilo); si no, uno
    propio POR HILO: el cliente de Supabase habla HTTP/2 por una sola conexión
    y compartida entre hilos se rompe («ConnectionTerminated», 3-oct)."""
    if _sb is not None:
        return _sb
    c = getattr(_por_hilo, "cliente", None)
    if c is None:
        from supabase import create_client
        url, clave = os.getenv("SUPABASE_URL", ""), os.getenv("SUPABASE_SERVICE_KEY", "")
        if not (url and clave):
            raise HTTPException(status_code=503, detail="Servicio temporalmente no disponible")
        c = _por_hilo.cliente = create_client(url, clave)
    return c


def cliente():
    """El cliente de Supabase con la clave de servicio (para los módulos que
    cobran: Bóveda, Cronología)."""
    return _cliente()


def _lista_admins() -> set:
    if _admins is not None:
        return _admins
    return _ADMIN_BASE | {e.strip().lower() for e in os.getenv("ADMIN_EMAILS", "").split(",") if e.strip()}


def consultas_por(unidades: int, por_consulta: int) -> int:
    """Cuántas consultas cuestan `unidades` (páginas, celdas…): hacia arriba.
    Cero unidades no cuestan nada; una sola ya cuesta una consulta."""
    if unidades <= 0:
        return 0
    return max(1, math.ceil(unidades / max(1, por_consulta)))


@dataclass
class Usuario:
    id: str
    email: str
    plan: Optional[str]
    es_admin: bool

    @property
    def es_platinum(self) -> bool:
        return self.es_admin or (self.plan or "") in PLANES_PLATINUM


async def usuario_del_token(authorization: Optional[str]) -> Usuario:
    """Quién llama. Manda el token, nunca un user_id del cuerpo."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Autenticación requerida")
    sb = _cliente()
    token = authorization.replace("Bearer ", "").strip()
    try:
        r = await asyncio.to_thread(sb.auth.get_user, token)
        u = r.user
        if not u:
            raise HTTPException(status_code=401, detail="Token inválido")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=401, detail="Token inválido o expirado")
    correo = (u.email or "").strip().lower()
    plan = None
    # UN TROPIEZO NO ES UNA FALTA DE PLAN: tres intentos antes de negar.
    for intento in (1, 2, 3):
        try:
            p = await asyncio.wait_for(asyncio.to_thread(
                lambda: sb.table("user_profiles").select("subscription_type")
                .eq("id", str(u.id)).limit(1).execute()), timeout=6.0)
            plan = (p.data[0].get("subscription_type") if p.data else None)
            break
        except Exception:
            if intento == 3:
                raise HTTPException(status_code=503, detail="No se pudo leer tu plan; vuelve a intentarlo.")
            await asyncio.sleep(0.4 * intento)
    return Usuario(id=str(u.id), email=correo, plan=plan, es_admin=correo in _lista_admins())


async def exigir_platinum(authorization: Optional[str]) -> Usuario:
    """El usuario si es Platinum (o Ultra, o administrador); si no, 403 con un
    código que el front usa para mostrar la invitación a mejorar el plan."""
    u = await usuario_del_token(authorization)
    if not u.es_platinum:
        raise HTTPException(status_code=403, detail={
            "codigo": "solo_platinum",
            "mensaje": "Esta función es exclusiva del plan Platinum.",
            "plan": u.plan,
        })
    return u


async def consultas_disponibles(user_id: str) -> dict:
    sb = _cliente()
    r = await asyncio.to_thread(lambda: sb.rpc("consultas_disponibles", {"p_user_id": user_id}).execute())
    return r.data or {"puede": False, "disponibles": 0}


async def cobrar_consultas(user_id: str, unidades: int, motivo: str) -> dict:
    """Cobra `unidades` consultas de golpe (todo o nada). Devuelve el resultado
    de la base: {allowed, movimiento_id, cobradas, disponibles, error?}.
    Si no alcanzan, `allowed` es False y NO se cobró nada."""
    if unidades <= 0:
        return {"allowed": True, "movimiento_id": None, "cobradas": 0}
    sb = _cliente()
    r = await asyncio.to_thread(lambda: sb.rpc("consumir_consultas", {
        "p_user_id": user_id, "p_unidades": int(unidades), "p_motivo": motivo[:300],
    }).execute())
    return r.data or {"allowed": False, "error": "sin_respuesta"}


async def devolver_consultas(movimiento_id: Optional[str], unidades: Optional[int] = None) -> dict:
    """Regresa hasta `unidades` (todas si es None) de un cobro. Idempotente."""
    if not movimiento_id or (unidades is not None and unidades <= 0):
        return {"devueltas": 0}
    sb = _cliente()
    params: dict[str, Any] = {"p_movimiento_id": movimiento_id}
    if unidades is not None:
        params["p_unidades"] = int(unidades)
    r = await asyncio.to_thread(lambda: sb.rpc("devolver_consultas", params).execute())
    return r.data or {"devueltas": 0}


def error_de_saldo(resultado: dict) -> HTTPException:
    """El 402 que ve el abogado cuando no le alcanza (o su cuenta está detenida)."""
    motivo = resultado.get("error") or "sin_saldo"
    mensajes = {
        "sin_saldo": "No te alcanzan las consultas para esto.",
        "suscripcion_suspendida": "Tu suscripción está suspendida por un pago pendiente.",
        "cuenta_bloqueada": "Tu cuenta está detenida.",
        "email_no_verificado": "Verifica tu correo para continuar.",
    }
    return HTTPException(status_code=402, detail={
        "codigo": motivo,
        "mensaje": mensajes.get(motivo, "No se pudo cobrar."),
        "disponibles": resultado.get("disponibles", 0),
        "requeridas": resultado.get("requeridas"),
    })
