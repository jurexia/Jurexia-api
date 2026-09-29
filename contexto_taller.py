# -*- coding: utf-8 -*-
"""EL CONTEXTO DE UNA PETICIÓN DEL TALLER: quién pide, qué banderas y qué se
excluye (rediseño del taller, punto 8 y «arregla el banco», 29-sep-2026).

POR QUÉ. Para medir cada cambio por separado (David, punto 8) hacen falta dos
cosas que el taller no tenía:

  1. LA EXCLUSIÓN DEL FALLO OBJETIVO. El banco Kingston corre contra
     producción, y el RAG podía traerle al motor la propia sentencia que se
     mide o lo que se decidió después: 18 de los 24 asuntos Kingston están en el
     índice de la OAJ (el 274/2025 con NEUN 38118729, fechado el 28-05-2026), y
     la única exclusión era el número tecleado. Aquí vive `Exclusion`: NEUN,
     expedientes (tipo, número, año), holdings, fecha de corte y serie, con
     predicados puros para cada fuente.
  2. BANDERAS POR PETICIÓN. Un cambio que toca los prompts de todos (la fuerza
     jurídica unificada, paso 1a) se mide encendido y apagado sobre las mismas
     sesiones antes de soltarlo. Las banderas se piden en la petición y sólo
     las acepta una cuenta de casa.

POR QUÉ UN `contextvars` Y NO UN GLOBAL NI UN PARÁMETRO. El API corre con
`gunicorn -w 2`: nada que deba sobrevivir a una petición puede vivir en memoria
del proceso ([[estado-entre-workers]]); por eso la exclusión y las banderas se
GUARDAN EN LA SESIÓN (`evaluacion` en `taller_sesiones.estado`) y cada petición
las vuelve a poner en su contexto al recuperar la sesión. Un `ContextVar` es
propio de cada petición —y de las tareas que ella lance, que lo heredan— así que
dos secretarios a la vez no se pisan. Pasarlo como parámetro obligaba a tocar
una docena de firmas entre la consulta y el estudio.

Fuera de una evaluación, el contexto está vacío y todo se comporta como antes:
`exclusion()` es None, y `bandera()` devuelve su valor por omisión.
"""
from __future__ import annotations

import contextvars
import datetime as _dt
import os
import re
from dataclasses import dataclass, field

_CTX: contextvars.ContextVar = contextvars.ContextVar("contexto_taller", default=None)


# ═══ LA EXCLUSIÓN ════════════════════════════════════════════════════════════

def fecha(x) -> _dt.date | None:
    """«28-05-2026», «2026-05-28», «2026-05-28T10:00:00» → date; None si no."""
    s = str(x or "").strip()
    if not s:
        return None
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        try:
            return _dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    m = re.match(r"^(\d{1,2})[-/](\d{1,2})[-/](\d{4})", s)
    if m:
        try:
            return _dt.date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        except ValueError:
            return None
    return None


def numero_anio(x):
    """(número, «año») de un expediente escrito como sea, o None."""
    m = re.search(r"(\d{1,5})\s*[/\-]\s*(\d{4})", str(x or ""))
    return (int(m.group(1)), m.group(2)) if m else None


def _neun(x):
    try:
        f = float(str(x).strip())
    except (TypeError, ValueError):
        return None
    return int(f) if f.is_integer() and f > 0 else None


@dataclass
class Exclusion:
    """Lo que el RAG NO puede traer mientras se evalúa un asunto.

    `expedientes`: números «N/AAAA» del fallo objetivo y de su serie (el tipo no
    se compara: con el número y el año, en el mismo tribunal, basta y sobra, y
    un tipo mal escrito no debe dejar pasar la fuga —la regla de `sin_fuga`—).
    `fecha_corte`: nada fechado en ese día o después (la sentencia objetivo; lo
    posterior revela el desenlace). Sin fecha legible, la fila NO se excluye por
    fecha —sólo por número o NEUN—: excluir lo que no se sabe fechar vaciaría el
    pozo sin razón.
    """
    neuns: set = field(default_factory=set)
    expedientes: set = field(default_factory=set)      # {(número, «año»)}
    holding_ids: set = field(default_factory=set)
    fecha_corte: _dt.date | None = None
    serie: str = ""
    web: bool = False                                   # la web no se puede cortar por fecha

    @classmethod
    def de_dict(cls, d: dict) -> "Exclusion | None":
        if not isinstance(d, dict):
            return None
        exps = set()
        for x in (d.get("expedientes") or []):
            na = numero_anio(x)
            if na:
                exps.add(na)
        neuns = {n for n in (_neun(x) for x in (d.get("neuns") or [])) if n}
        return cls(neuns=neuns, expedientes=exps,
                   holding_ids={str(x) for x in (d.get("holding_ids") or []) if str(x).strip()},
                   fecha_corte=fecha(d.get("fecha_corte")), serie=str(d.get("serie") or ""),
                   web=bool(d.get("web", False)))

    def a_dict(self) -> dict:
        return {"neuns": sorted(self.neuns),
                "expedientes": [f"{n}/{a}" for n, a in sorted(self.expedientes)],
                "holding_ids": sorted(self.holding_ids),
                "fecha_corte": self.fecha_corte.isoformat() if self.fecha_corte else "",
                "serie": self.serie, "web": self.web}

    def vacia(self) -> bool:
        return not (self.neuns or self.expedientes or self.holding_ids or self.fecha_corte)

    def _por_fecha(self, x) -> bool:
        f = fecha(x)
        return bool(self.fecha_corte and f and f >= self.fecha_corte)

    def excluye_fila(self, fila: dict) -> bool:
        """Una fila de la OAJ o del espejo viejo (payload o fila ya armada)."""
        if not isinstance(fila, dict):
            return False
        if _neun(fila.get("neun")) in self.neuns:
            return True
        for k in ("alias", "expediente", "numero"):
            if numero_anio(fila.get(k)) in self.expedientes:
                return True
        if str(fila.get("holding_id") or "") in self.holding_ids:
            return True
        return self._por_fecha(fila.get("fecha") or fila.get("fecha_sentencia"))

    def excluye_holding(self, payload: dict) -> bool:
        """Un holding o trozo de estudio de las colecciones de sentencias."""
        return self.excluye_fila(payload)

    def excluye_tesis(self, t: dict) -> bool:
        """Una tesis publicada en o después del corte (si trae fecha)."""
        if not isinstance(t, dict):
            return False
        return self._por_fecha(t.get("fecha_publicacion"))


# ═══ EL CONTEXTO DE LA PETICIÓN ══════════════════════════════════════════════

def poner(casa: bool = False, evaluacion: dict | None = None) -> None:
    """Pone el contexto de ESTA petición (y de las tareas que lance después).

    `evaluacion` = {"exclusion": {...}, "banderas": {...}} guardado en la
    sesión. Sólo se toma si la cuenta es de casa: un usuario no puede apagar
    fuentes ni encender banderas desde fuera."""
    ev = evaluacion if (casa and isinstance(evaluacion, dict)) else {}
    exc = Exclusion.de_dict(ev.get("exclusion") or {}) if ev.get("exclusion") else None
    if exc is not None and exc.vacia():
        exc = None
    ban = {str(k): v for k, v in (ev.get("banderas") or {}).items()} if isinstance(ev.get("banderas"), dict) else {}
    _CTX.set({"casa": bool(casa), "exclusion": exc, "banderas": ban})


def actual() -> dict:
    return _CTX.get() or {"casa": False, "exclusion": None, "banderas": {}}


def exclusion() -> Exclusion | None:
    return actual().get("exclusion")


def es_casa() -> bool:
    return bool(actual().get("casa"))


def bandera(nombre: str, defecto_env: str = "", omision: str = "casa") -> bool:
    """¿Está encendida la bandera `nombre` en esta petición?

    Manda, en este orden: la bandera pedida en la sesión de evaluación (sólo de
    casa); luego la variable de entorno `defecto_env` («todos» | «casa» | «0»);
    si no hay, `omision`. «casa» = sólo para las cuentas de casa: así un cambio
    que toca los prompts de todos se enciende primero donde se puede medir."""
    ban = actual().get("banderas") or {}
    if nombre in ban:
        return bool(ban[nombre])
    modo = (os.getenv(defecto_env, "") if defecto_env else "").strip().lower() or omision
    if modo in ("1", "true", "si", "sí", "todos"):
        return True
    if modo == "casa":
        return es_casa()
    return False
