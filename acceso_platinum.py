"""QUIÉN USA LAS HERRAMIENTAS DE PLATINUM.   (12-oct-2026)

David: «Todas estas herramientas bloquéalas a Pro y sólo déjaselas a
Platinum, con excepción de Jenica […], a quien su cuenta Pro debes
habilitarle todo. Igual los testers quedan con todo».

Las herramientas: Sentencia y Precedentes (viajan por /chat), Toulmin,
Jurimetría, Cronología, Bóveda y el escalón Platinum de la redacción. Pasan:
  · Platinum (mensual o anual) y Ultra Secretarios (`PLANES_PLATINUM`);
  · la administración (`ADMIN_EMAILS` de main.py, que se recibe de fuera);
  · la CORTESÍA: los testers (todo el dominio @iurexia.com) y Jenica
    (jenycampos@hotmail.com, cuenta pro_monthly).

LA CORTESÍA NO ES ADMINISTRACIÓN. `ADMIN_EMAILS` abre además el panel de
administración (`_verify_admin`), el piloto de la Corte IDH y la Bóveda sin
cobro. Meter ahí el dominio o a Jenica habría sido darles las llaves. Esta
lista sólo abre lo de Platinum, y se cobra como a cualquier Platinum.

EXIGE CORREO VERIFICADO. La confirmación de correo de Supabase estuvo apagada
desde mayo: cualquiera podía darse de alta como «x@iurexia.com», y el dominio
habría sido una puerta abierta a las herramientas más caras.
`email_verificado_at` sólo se llena al confirmar el correo (trigger
`al_confirmar_correo`, que no corre al dar de alta), al pagar o al entrar por
Google o Apple. Las 8 cuentas del dominio y la de Jenica lo tienen
(comprobado el 12-oct).

La web dice lo mismo en src/lib/herramientas-platinum.ts (`esAdmin` de
src/lib/admins.ts): si cambia una lista, cambia la otra.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Iterable, Optional

from fastapi import HTTPException

from esfuerzo_redaccion import PLANES_PLATINUM

DOMINIOS_CORTESIA = ("@iurexia.com",)          # los testers
CORREOS_CORTESIA = frozenset({
    "jenycampos@hotmail.com",                   # Jenica, pro_monthly
})


def _correo(c: Optional[str]) -> str:
    return (c or "").strip().lower()


def es_cortesia(correo: Optional[str], verificado: bool) -> bool:
    """Testers y Jenica, con el correo verificado."""
    c = _correo(correo)
    if not c or not verificado:
        return False
    return c in CORREOS_CORTESIA or c.endswith(DOMINIOS_CORTESIA)


def tiene_platinum(plan: Optional[str], correo: Optional[str] = None,
                   admin_emails: Iterable[str] = (), verificado: bool = False) -> bool:
    """Administración, cortesía o un plan Platinum."""
    c = _correo(correo)
    return (bool(c) and c in {_correo(e) for e in admin_emails}) \
        or es_cortesia(c, verificado) or (plan or "") in PLANES_PLATINUM


@dataclass(frozen=True)
class Acceso:
    plan: Optional[str]
    es_admin: bool      # sólo ADMIN_EMAILS: lo que no se cobra y el panel
    cortesia: bool
    leido: bool         # False = la base no respondió, no «no tiene plan»

    @property
    def platinum(self) -> bool:
        return self.es_admin or self.cortesia or (self.plan or "") in PLANES_PLATINUM


async def _leer(sb, campo: str, valor: str, admin_emails: Iterable[str]) -> Acceso:
    """El perfil por `campo`, con tres intentos. Un tropiezo de la base no es
    una falta de plan, pero tampoco abre la puerta: quien llama decide con
    `leido`."""
    if sb is None:
        return Acceso(None, False, False, False)

    def _perfil():
        return sb.table("user_profiles").select("subscription_type, email, email_verificado_at") \
            .eq(campo, valor).limit(1).execute()

    admins = {_correo(e) for e in admin_emails}
    for intento in (1, 2, 3):
        try:
            r = await asyncio.wait_for(asyncio.to_thread(_perfil), timeout=6.0)
            if not r.data:
                return Acceso(None, False, False, True)
            fila = r.data[0]
            c = _correo(fila.get("email"))
            return Acceso(fila.get("subscription_type"), bool(c and c in admins),
                          es_cortesia(c, bool(fila.get("email_verificado_at"))), True)
        except Exception as e:
            print(f"   ⚠️ acceso Platinum, intento {intento}/3: {type(e).__name__}")
            if intento < 3:
                await asyncio.sleep(0.4 * intento)
    return Acceso(None, False, False, False)


async def acceso_del_usuario(sb, user_id: Optional[str], admin_emails: Iterable[str] = ()) -> Acceso:
    """Sin usuario no hay nada que leer: se sabe que no tiene plan."""
    if not user_id:
        return Acceso(None, False, False, True)
    return await _leer(sb, "id", user_id, admin_emails)


async def acceso_por_correo(sb, correo: Optional[str], admin_emails: Iterable[str] = ()) -> Acceso:
    """Para Jurimetría mientras la web no mande el token (ver main.py)."""
    c = _correo(correo)
    if not c:
        return Acceso(None, False, False, True)
    return await _leer(sb, "email", c, admin_emails)


def rechazo(funcion: str = "Esta función", plan: Optional[str] = None) -> HTTPException:
    """El 403 que el front convierte en la invitación a mejorar el plan."""
    return HTTPException(403, {"codigo": "requiere_platinum",
                               "mensaje": f"{funcion} es exclusiva del plan Platinum.",
                               "plan": plan})


async def exigir(sb, user_id: Optional[str], admin_emails: Iterable[str] = (),
                 funcion: str = "Esta función") -> Acceso:
    """El acceso si es Platinum; 403 si no; 503 si no se pudo leer el plan:
    se cierra, no se abre."""
    a = await acceso_del_usuario(sb, user_id, admin_emails)
    if a.platinum:
        return a
    if not a.leido:
        raise HTTPException(503, "No se pudo comprobar tu plan. Vuelve a intentarlo en un momento.")
    raise rechazo(funcion, a.plan)
