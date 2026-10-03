# -*- coding: utf-8 -*-
"""LA PROPUESTA SIEMPRE ELIGE UN LADO: EL QUE PASA DEL 50% (David, 2-oct-2026).

«Si ya tenemos jurimetría y sólo hay dos sentidos (conceder o negar, fundado o
infundado), lo correcto es que si hay un 50.01% de probabilidad hacia un lado
sea esa la propuesta de resolución, para que el secretario pueda generar el
proyecto en automático. Esa función debe persistir desde un inicio.»

POR QUÉ NO BASTA CON EL VOTO DEL MOTOR. Medido en el banco Kingston con el oro
auditado sobre los resolutivos (21 amparos directos, 29-sep-2026):

    el motor votó CONCEDER en el 83% de los asuntos que el tribunal concedió
    y en el 71% de los que el tribunal NEGÓ.

Su voto casi no separa un desenlace del otro (razón de verosimilitud 1.17), y
la confianza que declara tampoco: «conceder, confianza alta» acertó 6 de 27.
El tribunal, en cambio, niega el 71% de sus amparos directos. Un número que
mezcla las tres fuentes con el peso que cada una DEMOSTRÓ es más honesto que
cualquiera de ellas sola, y es lo que aquí se calcula.

LAS TRES FUENTES, Y CÓMO PESA CADA UNA
 1. La TASA DEL TRIBUNAL por tipo de asunto (`probabilidad_sentido.json`,
    fichas OAJ): es el punto de partida, con el peso de `kappa` asuntos.
 2. Los PRECEDENTES DEL MISMO PROBLEMA (`fase_oaj.precedentes_oaj`, la tarjeta
    del principal): cada planteamiento del tribunal suma su calificación
    —prosperó o no— con el peso de su probabilidad calibrada de ser el mismo
    problema (0.85 o más para «mismo problema», 0.50-0.85 para «posible»). La
    coincidencia por TEMA del asunto pesa la mitad y se lee por el sentido de
    la sentencia, porque no trae calificación del planteamiento.
 3. El VOTO DEL MOTOR, con la razón de verosimilitud medida en Kingston. Se
    recalibra cuando el motor cambie: los números viven en el JSON.

LO QUE NO HACE
 · No escribe razones: si el lado que gana no es el que el motor razonó, se
   usa la vía contraria que el propio motor ya escribió «como si la
   defendiera» (`global.alternativa`, `checklist`).
 · No toca el sentido que haya dictado el secretario: corre sobre la
   propuesta, antes de que nadie decida.
 · Si no hay tasa para el tipo, decide el motor (no se inventa un número).
"""
from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

RUTA = Path(__file__).with_name("probabilidad_sentido.json")
_CAL: dict | None = None

# Las calificaciones que hacen PROSPERAR un planteamiento; las demás no. «Fundado
# pero insuficiente / pero inoperante» no prospera: tiene razón y no alcanza.
_RX_NO = re.compile(r"infundad|insuficient|inoperant|ineficaz|inatendibl|improceden", re.I)
_RX_SI = re.compile(r"fundad", re.I)
_RX_SIN = re.compile(r"no se estudi|sin materia|innecesari|desech|sobres", re.I)

# El sentido del resolutivo, para las filas por TEMA (sin calificación).
_RX_SENT_SI = re.compile(r"^\s*(ampara(?! ni)|revoca|modifica|fundad)", re.I)
_RX_SENT_NO = re.compile(r"^\s*(no ampara|confirma|infundad)", re.I)


def cargar(ruta: str | os.PathLike | None = None) -> dict:
    """La calibración; {} si no existe o no se lee (entonces decide el motor)."""
    global _CAL
    if ruta is None and _CAL is not None:
        return _CAL
    try:
        d = json.loads(Path(ruta or RUTA).read_text(encoding="utf-8"))
    except Exception as e:
        print(f"   ⚠️ probabilidad del sentido: sin calibración ({type(e).__name__})")
        d = {}
    if ruta is None:
        _CAL = d
    return d


def prospera(calificacion) -> int | None:
    """1 si la calificación hace prosperar el planteamiento, 0 si no, None si no
    dice nada (no se estudió, sin materia, vacía)."""
    s = " ".join(str(calificacion or "").replace("_", " ").split()).lower()
    if not s or _RX_SIN.search(s):
        return None
    if _RX_NO.search(s):
        return 0
    if _RX_SI.search(s):
        return 1
    return None


def _de_sentido(sentido) -> int | None:
    s = str(sentido or "")
    if _RX_SENT_NO.search(s):
        return 0
    if _RX_SENT_SI.search(s):
        return 1
    return None


def tasa(tipo: str, cal: dict | None = None) -> tuple:
    """(probabilidad de que prospere, asuntos que la sostienen) del tipo; (None, 0)
    si el tipo no está medido."""
    cal = cargar() if cal is None else cal
    t = (cal.get("tasas") or {}).get(str(tipo or "").strip().lower()) or {}
    try:
        p = float(t.get("prospera"))
    except (TypeError, ValueError):
        return None, 0
    if not 0.0 < p < 1.0:
        return None, 0
    return p, int(t.get("n") or 0)


def evidencia(filas: list, cal: dict | None = None) -> dict:
    """Los precedentes del principal como pseudo-cuentas ponderadas.

    Cada fila de `fase_oaj` trae `similitud` (probabilidad calibrada de ser el
    mismo problema, en %), `fuente` (planteamiento | tema), `calificacion` y
    `sentido`. Devuelve {a_favor, en_contra, n, filas: [...]} donde a_favor es la
    suma de pesos de los que prosperaron."""
    cal = cargar() if cal is None else cal
    peso_tema = float(cal.get("peso_tema", 0.5) or 0.5)
    a = b = 0.0
    usadas = []
    vistos = set()
    for f in filas or []:
        if not isinstance(f, dict):
            continue
        clave = (f.get("neun"), f.get("pregunta"))
        if clave in vistos:
            continue
        vistos.add(clave)
        try:
            w = max(0.0, min(0.99, float(f.get("similitud") or 0) / 100.0))
        except (TypeError, ValueError):
            continue
        if w < 0.5:
            continue
        if (f.get("fuente") or "planteamiento") == "tema":
            y = _de_sentido(f.get("sentido"))
            w *= peso_tema
        else:
            y = prospera(f.get("calificacion"))
        if y is None:
            continue
        if y:
            a += w
        else:
            b += w
        usadas.append({"expediente": f.get("expediente") or "", "similitud": f.get("similitud"),
                       "prospero": bool(y), "calificacion": f.get("calificacion") or f.get("sentido") or "",
                       "nivel": f.get("nivel") or ""})
    return {"a_favor": round(a, 3), "en_contra": round(b, 3), "n": len(usadas), "filas": usadas}


def voto_motor(sentido) -> int | None:
    """El voto del motor sobre si el asunto prospera (su sentido global)."""
    return prospera(sentido)


def calcular(tipo: str, filas: list | None = None, sentido_motor=None,
             cal: dict | None = None) -> dict:
    """La probabilidad de que el asunto PROSPERE y el lado que se propone.

    {p_prospera, lado: "prospera"|"no_prospera"|None, tasa, n_tasa, precedentes,
     motor, explicacion}. `lado` None sólo si no hay tasa para el tipo ni voto del
    motor: entonces no hay número que dar."""
    cal = cargar() if cal is None else cal
    p0, n0 = tasa(tipo, cal)
    ev = evidencia(filas or [], cal)
    v = voto_motor(sentido_motor)
    if p0 is None:
        # Sin tasa medida para el tipo no se inventa: decide el motor, como antes.
        lado = None if v is None else ("prospera" if v else "no_prospera")
        return {"p_prospera": None, "lado": lado, "tasa": None, "n_tasa": 0,
                "precedentes": ev, "motor": v, "fuente": "motor",
                "explicacion": "Sin tasa medida para este tipo de asunto: manda la propuesta del motor."}
    kappa = float(cal.get("kappa", 3) or 3)
    a = kappa * p0 + ev["a_favor"]
    b = kappa * (1.0 - p0) + ev["en_contra"]
    p = a / (a + b)
    lr = 1.0
    m = cal.get("motor") or {}
    if v is not None:
        try:
            lr = float(m.get("lr_prospera" if v else "lr_no_prospera") or 1.0)
        except (TypeError, ValueError):
            lr = 1.0
    odds = (p / (1.0 - p)) * max(lr, 1e-6)
    p = odds / (1.0 + odds)
    p = min(max(p, 0.01), 0.99)
    lado = "prospera" if p > 0.5 else "no_prospera"
    return {"p_prospera": round(p, 3), "lado": lado, "tasa": p0, "n_tasa": n0,
            "precedentes": ev, "motor": v, "fuente": "jurimetria",
            "explicacion": explicar(tipo, p, p0, n0, ev, v, lado)}


_NOMBRE_TIPO = {"amparo_directo": "amparo directo", "amparo_revision": "amparo en revisión",
                "queja": "queja", "revision_fiscal": "revisión fiscal"}
_PROSPERA_TXT = {"amparo_directo": ("conceder", "negar"),
                 "amparo_revision": ("que el recurso prospere", "que no prospere"),
                 "queja": ("que la queja sea fundada", "que sea infundada"),
                 "revision_fiscal": ("que el recurso prospere", "que no prospere")}


def _pct(x: float) -> str:
    return f"{int(round(x * 100))}%"


def explicar(tipo, p, p0, n0, ev, v, lado) -> str:
    """Una frase para el secretario: el número y de dónde sale."""
    si, no = _PROSPERA_TXT.get(tipo, ("que prospere", "que no prospere"))
    gana = si if lado == "prospera" else no
    prob = p if lado == "prospera" else 1 - p
    partes = [f"la tasa del tribunal en {_NOMBRE_TIPO.get(tipo, tipo)} ({_pct(p0)} prospera, {n0:,} asuntos)"]
    if ev["n"]:
        fav = sum(1 for f in ev["filas"] if f["prospero"])
        partes.append(f"{ev['n']} precedente{'s' if ev['n'] != 1 else ''} del tribunal sobre el mismo "
                      f"problema ({fav} en que prosperó y {ev['n'] - fav} en que no)")
    else:
        partes.append("ningún precedente del tribunal sobre el mismo problema")
    if v is not None:
        partes.append(f"la lectura del motor ({'apuntaba a que prospera' if v else 'apuntaba a que no prospera'}; "
                      f"su voto pesa poco: así lo midió el banco de 21 asuntos)")
    return (f"Se propone {gana}: probabilidad de {_pct(prob)}. Sale de " + ", ".join(partes[:-1])
            + (" y " if len(partes) > 1 else "") + partes[-1] + ".")
