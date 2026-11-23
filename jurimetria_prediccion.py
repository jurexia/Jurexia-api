# -*- coding: utf-8 -*-
"""PREDICCIÓN DEL SENTIDO, CON LO QUE SE MIDIÓ (23-nov-2026).

De dónde sale:
  · `datos/jurimetria_tasas.json.gz`: el listado oficial de la OAJ de los 7 circuitos con datos (2.2 M sentencias con
    resultado legible), contado por tribunal canónico × tipo × materia × ponente × etapa. El resultado se lee de la
    síntesis oficial de resolutivos; en amparo directo coincide con el sentido del SISE en el 99.8% (4,783).
  · `AJUSTES_REVISION` (aquí abajo): en amparo en revisión, lo que resolvió el juez de distrito y quién recurre. Regresión
    logística sobre el logit de la tasa, ajustada con 2,248 revisiones del 6.º circuito leídas de sus PDF
    (oaj/jurimetria/prediccion/origen.py).

Lo que se sabe de cada pieza (prueba hacia adelante: entrenado antes de 2025, medido en lo resuelto desde 2025):
  · Sólo tribunal, materia y ponente apenas pasan la tasa base (amparo directo: 61.3% de acierto contra 60.9% de
    decir siempre «niega»; AUC 0.62). Sirve para DESCRIBIR, no para adivinar un asunto.
  · En amparo en revisión, saber qué resolvió el juez y quién recurre lleva el acierto de 63.7% a 79.3% (AUC 0.82):
    el quejoso termina con amparo ~73% si el juez lo amparó, ~19% si lo negó, ~12% si sobreseyó.
  · En amparo directo, queja y revisión fiscal hace falta leer el asunto: esto no lo sustituye.

Puro y sin red: `predecir(...)` → dict con la probabilidad, la base, los ajustes explicados y la advertencia honesta.
"""
from __future__ import annotations

import gzip
import json
import math
import os
from collections import defaultdict

import jurimetria_canon as jc

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datos")
M = 30                       # encogimiento: cuántas sentencias «vale» el nivel de arriba
MIN_PROPIA = 30              # con menos, se dice que la base es prestada
AJUSTES_REVISION = {"logit_tasa": 1.7617, "juez=AMPARA": 2.0307, "juez=AMPARA_PARCIAL": 0.7298, "juez=NIEGA": -0.2478, "juez=SOBRESEE": -0.8192, "juez=DESECHA": 0.0661, "recurre=quejoso": -0.2692, "recurre=tercero": 0.7403, "recurre=autoridad": 0.4842, "incidental": 0.2663, "intercepto": 0.3795}



def _norm(s) -> str:
    return " ".join(jc.plano(s or "").split())


class _Tablas:
    def __init__(self):
        self.ok = False
        self.ponente = defaultdict(lambda: [0, 0])    # (trib, tipo, mat, pon) → [n, k]
        self.materia = defaultdict(lambda: [0, 0])    # (trib, tipo, mat)
        self.tribunal = defaultdict(lambda: [0, 0])   # (trib, tipo)
        self.tipo = defaultdict(lambda: [0, 0])       # (tipo,)

    def cargar(self, etapa: str = "todas"):
        ruta = os.path.join(_DIR, "jurimetria_tasas.json.gz")
        if not os.path.exists(ruta):
            return self
        with gzip.open(ruta, "rt", encoding="utf-8") as g:
            conteos = json.load(g)["conteos"]
        for clave, (n, k) in conteos.items():
            trib, tipo, mat, pon, et = clave.split("|")
            if etapa != "todas" and et != etapa:
                continue
            for d, kk in ((self.ponente, (trib, tipo, mat, pon)), (self.materia, (trib, tipo, mat)),
                          (self.tribunal, (trib, tipo)), (self.tipo, (tipo,))):
                d[kk][0] += n
                d[kk][1] += k
        self.ok = bool(self.tipo)
        return self


_TABLAS: dict[str, _Tablas] = {}


def tablas(etapa: str = "todas") -> _Tablas:
    if etapa not in _TABLAS:
        _TABLAS[etapa] = _Tablas().cargar(etapa)
    return _TABLAS[etapa]


def wilson(k: float, n: float, z: float = 1.96):
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round((c - m) / d, 3), round((c + m) / d, 3)]


def tasa(tribunal: str | None, tipo: str, materia: str | None = None, ponente: str | None = None,
         etapa: str = "todas") -> dict | None:
    """La tasa con la que prospera lo que pide quien promueve, de lo general a lo particular, encogida (m=30) hacia el
    nivel de arriba: tipo → tribunal → tribunal×materia → ponente. Dice cuántas sentencias PROPIAS la sostienen."""
    T = tablas(etapa)
    tipo = jc.tipo_canon(tipo) or tipo
    if not T.ok or (tipo,) not in T.tipo:
        return None
    n, k = T.tipo[(tipo,)]
    p, nivel, propia = k / n, "nacional (7 circuitos)", (n, k)
    if tribunal:
        n1, k1 = T.tribunal.get((tribunal, tipo), (0, 0))
        if n1:
            p, nivel, propia = (k1 + M * p) / (n1 + M), "tribunal", (n1, k1)
            mat = jc.materia_canon(materia)[0] if materia else None
            n2, k2 = T.materia.get((tribunal, tipo, mat), (0, 0)) if mat else (0, 0)
            if n2:
                p, nivel, propia = (k2 + M * p) / (n2 + M), "tribunal y materia", (n2, k2)
                pon = _norm(ponente)
                n3, k3 = T.ponente.get((tribunal, tipo, mat, pon), (0, 0)) if pon else (0, 0)
                if n3:
                    p, nivel, propia = (k3 + M * p) / (n3 + M), "ponente", (n3, k3)
    return {"p": round(p, 4), "nivel": nivel, "n": propia[0], "prosperan": propia[1],
            "wilson95": wilson(propia[1], propia[0]), "prestada": propia[0] < MIN_PROPIA}


_JUEZ = {"AMPARA": "el juez de distrito concedió el amparo", "AMPARA_PARCIAL": "el juez concedió en parte",
         "NIEGA": "el juez negó el amparo", "SOBRESEE": "el juez sobreseyó", "DESECHA": "el juez desechó"}
_RECURRE = {"quejoso": "recurre el quejoso", "tercero": "recurre el tercero interesado",
            "autoridad": "recurre la autoridad"}


def juez_canon(j) -> str | None:
    t = _norm(j)
    if not t:
        return None
    if "parcial" in t or "en parte" in t:
        return "AMPARA_PARCIAL"
    if t.startswith(("nego", "niega", "neg")) or "no ampara" in t:
        return "NIEGA"
    if "sobres" in t:
        return "SOBRESEE"
    if "desech" in t:
        return "DESECHA"
    if "ampar" in t or "conced" in t or "otorg" in t:
        return "AMPARA"
    return None


def recurre_canon(r) -> str | None:
    t = _norm(r)
    if not t:
        return None
    if "quejos" in t:
        return "quejoso"
    if "tercer" in t:
        return "tercero"
    if "autoridad" in t or "ministerio" in t or "fiscal" in t:
        return "autoridad"
    return None


def predecir(tribunal: str | None, tipo: str, materia: str | None = None, ponente: str | None = None,
             juez: str | None = None, recurre: str | None = None, incidental: bool = False) -> dict:
    """Probabilidad de que prospere lo que pide quien promueve (en amparo: que el quejoso salga con amparo)."""
    tipo_c = jc.tipo_canon(tipo) or tipo
    base = tasa(tribunal, tipo_c, materia, ponente)
    if not base:
        return {"probabilidad": None, "advertencia": "SIN BASE SUFICIENTE"}
    out = {"probabilidad": base["p"], "base": base, "ajustes": [], "tipo": tipo_c}
    aj = AJUSTES_REVISION if tipo_c == "amparo_revision" else None
    j, q = juez_canon(juez), recurre_canon(recurre)
    if aj and (j or q):
        pb = min(max(base["p"], 1e-3), 1 - 1e-3)
        z = aj["intercepto"] + aj["logit_tasa"] * math.log(pb / (1 - pb))
        if j:
            z += aj.get(f"juez={j}", 0.0)
            out["ajustes"].append(_JUEZ[j])
        if q:
            z += aj.get(f"recurre={q}", 0.0)
            out["ajustes"].append(_RECURRE[q])
        if incidental:
            z += aj.get("incidental", 0.0)
            out["ajustes"].append("es revisión incidental (suspensión)")
        out["probabilidad"] = round(1 / (1 + math.exp(-z)), 4)
        out["precision_medida"] = "79% de acierto en revisiones resueltas en 2025-26 (AUC 0.82)"
    elif tipo_c == "amparo_revision":
        out["advertencia"] = ("Sin saber qué resolvió el juez y quién recurre, esto es sólo la estadística del tribunal "
                              "(acierta poco más que la tasa base). Con esos dos datos el acierto medido sube a 79%.")
    else:
        out["advertencia"] = ("Es la estadística del tribunal: describe cómo resuelve, no adivina este asunto. En "
                              "amparo directo, queja y revisión fiscal la predicción depende de leer el asunto.")
    return out
