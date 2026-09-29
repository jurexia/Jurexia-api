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
        """Tolerante con lo que escribe una persona: un valor suelto vale como
        lista de uno; lo que no se entiende se descarta (nunca lanza)."""
        if not isinstance(d, dict):
            return None

        def _lista(v):
            if v is None:
                return []
            if isinstance(v, (list, tuple, set)):
                return list(v)
            return [v]
        d = {k: (_lista(v) if k in ("neuns", "expedientes", "holding_ids") else v) for k, v in d.items()}
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

def poner(casa: bool = False, evaluacion: dict | None = None, pruebas: bool | None = None) -> None:
    """Pone el contexto de ESTA petición (y de las tareas que lance después).

    `evaluacion` = {"exclusion": {...}, "banderas": {...}} guardado en la
    sesión. Sólo se toma si la cuenta es de casa: un usuario no puede apagar
    fuentes ni encender banderas desde fuera."""
    ev = evaluacion if (casa and isinstance(evaluacion, dict)) else {}
    try:
        exc = Exclusion.de_dict(ev.get("exclusion") or {}) if ev.get("exclusion") else None
    except Exception:
        exc = None
    if exc is not None and exc.vacia():
        exc = None
    ban = {}
    if isinstance(ev.get("banderas"), dict):
        for k, v in ev["banderas"].items():
            b = _booleano(v)
            if b is not None:
                ban[str(k)] = b
    # LAS CUENTAS DE PRUEBA (David, 29-sep-2026: «en donde voy a probar el
    # redactor es en las cuentas de @iurexia.com»). Las banderas «casa» se
    # encienden para ellas; su cuenta personal (jdm.juridico, que es de casa
    # por ADMIN_EMAILS) queda como la de cualquier secretario. Sin decirlo,
    # prueba = casa.
    _CTX.set({"casa": bool(casa), "pruebas": bool(casa if pruebas is None else pruebas),
              "exclusion": exc, "banderas": ban})


def _booleano(v):
    """True/False de un valor escrito a mano («false», «0», «no» son False);
    None si no se entiende (entonces no se usa y manda el entorno)."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v or "").strip().lower()
    if s in ("1", "true", "si", "sí", "on", "encendida", "yes"):
        return True
    if s in ("0", "false", "no", "off", "apagada"):
        return False
    return None


def valida(evaluacion) -> str:
    """«» si la evaluación se puede usar; si no, qué tiene mal (para un 422
    ANTES de generar, no un 500 después)."""
    if not isinstance(evaluacion, dict):
        return "la evaluación debe ser un objeto JSON"
    exc = evaluacion.get("exclusion")
    if exc is not None:
        if not isinstance(exc, dict):
            return "«exclusion» debe ser un objeto"
        e = Exclusion.de_dict(exc)
        if e is None or e.vacia():
            return "la exclusión quedó vacía: faltan NEUN, expedientes, holdings o fecha de corte legibles"
        if exc.get("fecha_corte") and e.fecha_corte is None:
            return f"fecha_corte ilegible: {exc.get('fecha_corte')!r}"
    ban = evaluacion.get("banderas")
    if ban is not None and not isinstance(ban, dict):
        return "«banderas» debe ser un objeto"
    for k, v in (ban or {}).items():
        if _booleano(v) is None:
            return f"la bandera {k!r} no es sí/no: {v!r}"
    return ""


def filtrar_tesis(tesis) -> list:
    """Las tesis sin lo que la evaluación excluye (publicadas desde el corte).
    Fuera de una evaluación, la lista tal cual. UN SOLO FILTRO para todas las
    entradas de tesis: consulta, figura, refuerzo, citadas y tardías."""
    exc = exclusion()
    lista = list(tesis or [])
    if exc is None:
        return lista
    return [t for t in lista if not exc.excluye_tesis(t)]


def aplicado() -> dict:
    """Lo que RIGIÓ en esta petición, para devolverlo a una cuenta de casa: el
    banco comprueba con esto que el servidor aplicó lo que se le pidió (un
    worker con código viejo ignora los campos sin error)."""
    c = actual()
    exc = c.get("exclusion")
    return {"exclusion": exc.a_dict() if exc is not None else None,
            "banderas": dict(c.get("banderas") or {})}


def actual() -> dict:
    return _CTX.get() or {"casa": False, "pruebas": False, "exclusion": None, "banderas": {}}


def es_de_pruebas() -> bool:
    return bool(actual().get("pruebas"))


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
    if nombre in ban and _booleano(ban[nombre]) is not None:
        return _booleano(ban[nombre])
    modo = (os.getenv(defecto_env, "") if defecto_env else "").strip().lower() or omision
    if modo in ("1", "true", "si", "sí", "todos"):
        return True
    if modo == "casa":
        # «casa» para las banderas = las cuentas DE PRUEBA (ver `poner`).
        return es_de_pruebas()
    return False


# ═══ LAS BANDERAS DEL REDISEÑO (revisión del 29-sep) ═════════════════════════
# Cada cambio que altera lo que ve o recibe un secretario va detrás de la suya,
# «casa» por omisión: la cuenta de David lo ve, los demás no, hasta medirlo con
# el banco (base con todas apagadas = producción de hoy para los de fuera).
BANDERAS_REDISENO = {
    "fuerza_unificada": "FUERZA_UNIFICADA",          # fuerza por tribunal, clave, vigencia única
    "fuente_tardia_aviso": "FUENTE_TARDIA_AVISO",    # aviso «justificación pendiente»
    "normas_al_documento": "NORMAS_AL_DOCUMENTO",    # artículos recuperados al .docx (con litis)
    "tesis_parte_al_consultar": "TESIS_PARTE_AL_CONSULTAR",  # decisión 3
    "consulta_provisional": "CONSULTA_PROVISIONAL",  # la tarjeta no recomienda sin la figura
    "analisis_neutral": "ANALISIS_NEUTRAL",          # etapa 2: análisis de la litis antes de proponer
    "recuperacion_requisitos": "RECUPERACION_REQUISITOS",  # etapa 2: búsqueda por requisito
    "propuesta_unica": "PROPUESTA_UNICA",            # etapa 3, paso 1: una sola propuesta viva por adelanto
    "plan_con_analisis": "PLAN_CON_ANALISIS",        # etapa 4: alias, cambios sin justificar, sin_clasificar, P del análisis
    "exclusiones_con_prueba": "EXCLUSIONES_CON_PRUEBA",  # etapa 4: registro común de exclusión con su prueba
    "estados_sesion": "ESTADOS_SESION",              # etapa 4: manifiesto «consume» y estado de la sesión
    "soluciones_por_desenlace": "SOLUCIONES_POR_DESENLACE",  # etapa 3: N justificadores, uno por solución
    "revision_semantica": "REVISION_SEMANTICA",      # etapa 3: la propuesta revisada por código (sólo avisa)
}


# LAS QUE NACEN APAGADAS TAMBIÉN PARA CASA. `soluciones_por_desenlace` no pasó
# sus compuertas en el banco de deliberación (29-sep-2026: exactitud 0.38 y
# «claro» acertado 1 de 4, compuerta 0.80): encendida para las cuentas de casa,
# la tarjeta recomendaría con una certeza que no tiene. Sigue medible: la
# sesión de evaluación la pide por su nombre, o su variable de entorno la abre.
OMISION_REDISENO = {"soluciones_por_desenlace": "0"}


def rediseno(nombre: str) -> bool:
    """¿Rige este cambio del rediseño en esta petición? (ver BANDERAS_REDISENO)."""
    return bandera(nombre, BANDERAS_REDISENO.get(nombre, ""), OMISION_REDISENO.get(nombre, "casa"))
