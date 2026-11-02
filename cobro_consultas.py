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
import freno_estudio

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
    # `usadas` (2-nov-2026): el `queries_used` de esa misma lectura, para el
    # freno del plan Estudio sin otro viaje a la base (freno_estudio.py).
    return {"plan": a.plan, "es_admin": a.es_admin, "cortesia": a.cortesia, "usadas": a.usadas}


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
    # PLAN ESTUDIO (2-nov-2026). Bóveda y Cronología sólo corren dentro de sus
    # 560 consultas de élite del periodo (freno_estudio.py), así que se cotiza
    # contra ésas y no contra el tope de 3,100. Las recargas NO cuentan aquí:
    # `consumir_consultas` cobra primero del plan, y con 3,100 de tope una
    # tarea de Bóveda pagaría con consultas del plan aunque se le dijera que
    # pagaba con la recarga. Que una recarga abra la Bóveda después de las 560
    # pide cambiar esa función (precios-nov2/estudio-hoy/api/decisiones.md).
    # `freno` lleva lo que queda de las 560 y, agotadas, el aviso para que la
    # pantalla lo diga antes de que el abogado suba nada. Sin contador legible
    # sigue abajo, como cualquier plan con tope (como Platinum).
    elite = freno_estudio.elite_restantes(f.get("subscription_type"), f.get("queries_used"))
    if elite is None and freno_estudio.es_estudio(f.get("subscription_type")):
        freno_estudio.sin_contador(f.get("queries_used"), "saldo")
    if elite is not None:
        return {"ilimitado": False, "del_plan": elite, "recargadas": recargadas,
                "disponibles": elite, "suspendida": bool(f.get("suspendido_at")),
                "freno": freno_estudio.resumen(f.get("subscription_type"), f.get("queries_used"))}
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
                           referencia: Optional[str] = None, plan: Optional[str] = None,
                           usadas: Optional[int] = None) -> dict:
    """Cobra `unidades` consultas de una vez, o ninguna.

    Devuelve {cobro_id, unidades, de_plan, de_recarga, restantes, ...}.
    Lanza 402 si no alcanza (con `necesarias` y `disponibles`), 403 si la
    cuenta no puede consultar, 503 si la base no respondió: SIN COBRO NO SE
    PROCESA, al revés que el chat, porque aquí lo que se arriesga es el
    expediente entero.

    `plan` y `usadas` (2-nov-2026): quien llama ya los leyó al pasar la
    puerta (acceso_platinum.py los trae en la misma lectura). Con Estudio,
    antes de cobrar se comprueba que la tarea quepa en lo que le queda de sus
    560 consultas de élite; si no, 402 `freno_estudio` y no se cobra nada.
    Sólo si no llegan `usadas` se lee el contador; los demás planes no pagan
    ninguna lectura más. Si el contador no se puede leer, se cobra como a
    Platinum (decisión del 2-nov) y queda la línea «SIN CONTADOR»."""
    if freno_estudio.es_estudio(plan):
        if usadas is None:
            try:
                r = await asyncio.to_thread(
                    lambda: sb.table("user_profiles").select("queries_used")
                    .eq("id", user_id).limit(1).execute())
                usadas = (r.data or [{}])[0].get("queries_used")
            except Exception as e:
                print(f"   ⚠️ freno de Estudio: no pude leer el contador ({type(e).__name__})")
        try:
            freno_estudio.exigir_elite(plan, usadas, int(unidades), donde=motivo)
        except HTTPException:
            print(f"   🧭 freno de Estudio: {motivo} pide {unidades} y le quedan "
                  f"{freno_estudio.elite_restantes(plan, usadas)} de "
                  f"{freno_estudio.CONSULTAS_ELITE} de élite — no se cobra")
            raise

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
        # Estudio (2-nov-2026): lo que la pantalla muestra como «te quedan» es
        # lo que queda de las 560 de élite, que es lo que la Bóveda y la
        # Cronología pueden gastar; contra el tope de 3,100 diría miles. El
        # número del tope sigue en `restantes_del_tope`.
        elite = freno_estudio.elite_restantes(plan, usadas)
        if elite is not None:
            d = {**d, "restantes_del_tope": d.get("restantes"),
                 "restantes": max(0, elite - int(unidades))}
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
