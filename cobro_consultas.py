"""COBRO PROPORCIONAL DE CONSULTAS Y PUERTA PLATINUM.   (12-oct-2026)

David: «esto consumirá las consultas en proporción, sino el usuario con
expedientes completos rebasaría el costo en servicios que lo que paga por su
plan, desde luego que esto es exclusivo de platinum».

Lo comparten la Cronología del expediente y la Bóveda: las dos leen
expedientes enteros, las dos son de Platinum y las dos cobran por páginas.
Vive fuera de main.py para que ninguna de las dos lo duplique.

EL MISMO CONTADOR QUE EL CHAT. `consumir_consultas` (Supabase, ver
migration_cobro_consultas.sql) descuenta de `queries_used` y, agotada la
cuota del plan, de `consultas_recargadas`: lo mismo que `consume_query`, pero
N de una vez y todo o nada. Cada cobro deja fila en `consultas_cobros`, y
`devolver_cobro` devuelve exactamente lo cobrado si el trabajo falla.

LA PUERTA SE COMPRUEBA EN EL SERVIDOR. Que la pantalla esconda el botón no
basta: cualquiera puede llamar al endpoint. Mismo criterio que la redacción
Platinum del chat (`_plan_para_redaccion` + `PLANES_PLATINUM`), y si el plan
no se puede leer, NO se abre.
"""
from __future__ import annotations

import asyncio
import math
import os
from typing import Optional

from fastapi import HTTPException

import acceso_platinum

# 1 consulta por cada N páginas, mínimo 1. Ver el informe de costos de la
# Cronología (12-oct-2026): N=10 deja el margen del plan, contando OCR.
PAGINAS_POR_CONSULTA = max(1, int(os.getenv("PAGINAS_POR_CONSULTA", "10")))


def consultas_por_paginas(paginas: int, n: Optional[int] = None) -> int:
    """max(1, ⌈páginas / N⌉)."""
    n = max(1, int(n or PAGINAS_POR_CONSULTA))
    return max(1, math.ceil(max(0, int(paginas or 0)) / n))


async def plan_del_usuario(sb, user_id: str, admin_emails=()) -> tuple:
    """(subscription_type, es_admin, leído), con tres intentos.

    Un tropiezo de la base no es una falta de plan, pero tampoco abre la
    puerta: quien llama decide con `leído`. La lectura es la de
    acceso_platinum.py, que además sabe de la cortesía."""
    if not user_id or sb is None:
        return None, False, False
    a = await acceso_platinum.acceso_del_usuario(sb, user_id, admin_emails)
    return a.plan, a.es_admin, a.leido


async def exigir_platinum(sb, user_id: str, admin_emails=(), funcion: str = "Esta función") -> dict:
    """403 si no es Platinum (ni Ultra, ni administración, ni cortesía: los
    testers y Jenica, ver acceso_platinum.py). 503 si no se pudo leer el
    plan: se cierra, no se abre."""
    a = await acceso_platinum.exigir(sb, user_id, admin_emails, funcion)
    return {"plan": a.plan, "es_admin": a.es_admin, "cortesia": a.cortesia}


async def saldo_consultas(sb, user_id: str) -> dict:
    """Lo que le queda para cotizar ANTES de procesar. No cobra nada: el cobro
    de verdad lo decide `consumir_consultas` bajo cerrojo."""
    def _leer():
        return sb.table("user_profiles") \
            .select("queries_used, queries_limit, consultas_recargadas, suspendido_at, subscription_type") \
            .eq("id", user_id).limit(1).execute()

    r = await asyncio.to_thread(_leer)
    if not r.data:
        return {"ilimitado": False, "del_plan": 0, "recargadas": 0, "disponibles": 0}
    f = r.data[0]
    limite = int(f.get("queries_limit") or 0)
    usadas = int(f.get("queries_used") or 0)
    recargadas = int(f.get("consultas_recargadas") or 0)
    if limite == -1:
        return {"ilimitado": True, "del_plan": None, "recargadas": recargadas, "disponibles": None,
                "suspendida": bool(f.get("suspendido_at"))}
    del_plan = max(limite - usadas, 0)
    return {"ilimitado": False, "del_plan": del_plan, "recargadas": recargadas,
            "disponibles": del_plan + recargadas, "suspendida": bool(f.get("suspendido_at"))}


_MENSAJES = {
    "saldo_insuficiente": "No te alcanzan las consultas: hacen falta {necesarias} y te quedan {disponibles}.",
    "cuenta_bloqueada": "Tu cuenta está bloqueada.",
    "suscripcion_suspendida": "Tu suscripción está suspendida por un pago pendiente. Actualiza tu tarjeta para continuar.",
    "email_no_verificado": "Verifica tu correo para continuar.",
}


async def cobrar_consultas(sb, user_id: str, unidades: int, motivo: str,
                           referencia: Optional[str] = None) -> dict:
    """Cobra `unidades` consultas de una vez, o ninguna.

    Devuelve {cobro_id, unidades, de_plan, de_recarga, restantes, ...}.
    Lanza 402 si no alcanza (con `necesarias` y `disponibles`), 403 si la
    cuenta no puede consultar, 503 si la base no respondió: SIN COBRO NO SE
    PROCESA, al revés que el chat, porque aquí lo que se arriesga es el
    expediente entero."""
    def _rpc():
        return sb.rpc("consumir_consultas", {"p_user_id": user_id, "p_unidades": int(unidades),
                                              "p_motivo": motivo, "p_referencia": referencia}).execute()
    try:
        r = await asyncio.to_thread(_rpc)
    except Exception as e:
        print(f"   ❌ cobrar_consultas({motivo}, {unidades}) falló: {type(e).__name__}: {str(e)[:200]}")
        raise HTTPException(503, "No se pudo registrar el cobro. Vuelve a intentarlo; no se descontó nada.")
    d = r.data or {}
    if d.get("allowed"):
        print(f"   🧾 cobro {motivo}: {unidades} consultas ({d.get('de_plan')} del plan, "
              f"{d.get('de_recarga')} de recarga) · cobro {str(d.get('cobro_id'))[:8]}")
        return d
    codigo = d.get("error") or "rechazado"
    mensaje = _MENSAJES.get(codigo, "No se pudo cobrar la consulta.").format(
        necesarias=d.get("necesarias", unidades), disponibles=d.get("disponibles", 0))
    raise HTTPException(402 if codigo == "saldo_insuficiente" else 403,
                        {"codigo": codigo, "mensaje": mensaje,
                         "necesarias": d.get("necesarias", unidades),
                         "disponibles": d.get("disponibles")})


async def devolver_cobro(sb, cobro_id: Optional[str], unidades: Optional[int] = None) -> dict:
    """Devuelve lo cobrado (todo, o `unidades`). Nunca lanza: una devolución
    que falla se registra y se reintenta a mano, no tumba la respuesta."""
    if not cobro_id or sb is None:
        return {"devuelta": False, "error": "sin_cobro"}
    try:
        params = {"p_cobro_id": str(cobro_id)}
        if unidades is not None:
            params["p_unidades"] = int(unidades)
        r = await asyncio.to_thread(lambda: sb.rpc("devolver_consultas", params).execute())
        d = r.data or {}
        print(f"   ↩️ devolución del cobro {str(cobro_id)[:8]}: {d}")
        return d
    except Exception as e:
        print(f"   🚨 NO SE PUDO DEVOLVER el cobro {cobro_id}: {type(e).__name__}: {str(e)[:200]}")
        return {"devuelta": False, "error": "fallo"}
