# -*- coding: utf-8 -*-
"""EL REINTENTO DEL CHAT, CONTADO (30-sep-2026).

Cuando la primera petición del chat falla del lado del abogado —la red del
teléfono, casi siempre— el servidor nunca se entera. En una semana de registros
de Render no hubo un solo error del chat, y aun así David vio el aviso de
reintento. Desde ahora el cliente manda en el reintento por qué reintenta
(`ChatRequest.reintento`) y aquí se vuelve UNA línea de registro que se cuenta:

    GET /v1/logs?text=REINTENTO_CHAT

Son datos del navegador: van acotados, sin saltos de línea y sin nada del
abogado (sólo el intento, el tipo de falla, el status y un texto de error corto).
"""
from __future__ import annotations

import re
from typing import Any

MARCA = "REINTENTO_CHAT"


def _num(v: Any, tope: int) -> int:
    try:
        return max(0, min(int(v), tope))
    except (TypeError, ValueError):
        return 0


def linea(r: Any) -> str:
    """La línea de registro, o «» si lo recibido no es un reintento legible."""
    if not isinstance(r, dict):
        return ""
    tipo = re.sub(r"[^a-z_]", "", str(r.get("tipo") or "")[:20].lower()) or "desconocido"
    error = " ".join(str(r.get("error") or "").split())[:80]
    return (f"🔁 {MARCA} intento={_num(r.get('intento'), 10)} tipo={tipo} "
            f"status={_num(r.get('status'), 999)} espera_ms={_num(r.get('espera_ms'), 60_000)} "
            f"error={error!r}")
