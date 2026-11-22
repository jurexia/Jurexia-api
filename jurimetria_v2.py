# -*- coding: utf-8 -*-
"""LA JURIMETRÍA QUE NO MIENTE (22-nov-2026, fase 0 del plan aprobado por David).

Lo que hacía `_aggregate_jurimetria` y por qué se retira (medido en el censo de `sentencias_holdings`):
  · Con 0 resultados —la materia «administrativo» que manda la web, un circuito sin datos— contestaba «NIEGA»
    con confianza baja: una predicción inventada.
  · Contaba los 80 más parecidos, de cualquier tipo de asunto, como una sola población: amparo directo, revisión y
    queja tienen tasas de éxito muy distintas y mezclarlas no predice ninguno.
  · «confirma» → NIEGA, «modifica» → SOBRESEE, sin decir desde el punto de vista de quién.

Lo que hace esto, con los campos canónicos `j_*` (jurimetria_canon.py):
  · UNA sentencia, UN voto (`j_sentencia`).
  · La base es «el mismo problema» (similitud ≥ UMBRAL_MISMO, el umbral que midió SIAPJF para text-embedding-3-small);
    si no alcanza, los más parecidos, y se dice.
  · Se separa por tipo de asunto: manda el que pidió el usuario o el que domina en la base.
  · La pregunta es `j_prospera`: ¿prosperó lo que pidió quien promovió (el quejoso en amparo directo, el recurrente
    en un recurso)? Y aparte, cuando se sabe, `j_quejoso`: cómo quedó el quejoso o el particular.
  · El circuito pide prestado a lo nacional cuando su base es corta (encogimiento: (k + m·p_nac)/(n + m)), y el
    intervalo de Wilson dice cuánto se sabe de verdad.
  · Sin base: «SIN BASE SUFICIENTE», nunca un sentido.

Puro: recibe puntos (dicts con `score` y `payload`) y devuelve el dict de `estadistica`. La búsqueda vive en main.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

import jurimetria_canon as jc

UMBRAL_MISMO = 0.73          # coseno: «el mismo problema jurídico» (SIAPJF, contrato.UMBRAL_MISMO_PROBLEMA)
MIN_MISMO = 12               # con menos, la base pasa a «los más parecidos» y se avisa
TOPE_PARECIDOS = 60
PESO_NACIONAL = 10           # m del encogimiento: cuántas sentencias «vale» lo nacional dentro del circuito
SIN_BASE = "SIN BASE SUFICIENTE"

# La pantalla (JurimetriaModal.tsx) conoce estas tres claves: «Concede», «Niega», «Sobresee / Sin materia».
# Leídas desde quien promovió: prosperó (o en parte) / no prosperó / no se entró al fondo.
LEGADO = {"si": "CONCEDE", "parcial": "CONCEDE", "no": "NIEGA", "sin_fondo": "SOBRESEE"}
NOMBRE_PROSPERA = {"si": "prospera", "parcial": "prospera en parte", "no": "no prospera",
                   "sin_fondo": "no se entra al fondo"}


def _pl(p):
    return (p.get("payload") if isinstance(p, dict) else getattr(p, "payload", None)) or {}


def _sc(p):
    s = p.get("score") if isinstance(p, dict) else getattr(p, "score", None)
    try:
        return float(s or 0.0)
    except (TypeError, ValueError):
        return 0.0


def una_por_sentencia(puntos: list) -> list:
    """Cada sentencia una vez, con su mejor similitud. Sin `j_sentencia`, el id del punto."""
    mejor = {}
    for p in puntos or []:
        pl = _pl(p)
        k = pl.get("j_sentencia") or (p.get("id") if isinstance(p, dict) else getattr(p, "id", None)) or id(p)
        if k not in mejor or _sc(p) > _sc(mejor[k]):
            mejor[k] = p
    return sorted(mejor.values(), key=_sc, reverse=True)


def base_de(puntos: list) -> tuple[list, str]:
    """(base, clase): «mismo_problema» si hay al menos MIN_MISMO con similitud ≥ UMBRAL_MISMO; si no, los
    TOPE_PARECIDOS más parecidos («parecidos»); «sin_base» si no hay nada."""
    u = una_por_sentencia(puntos)
    mismo = [p for p in u if _sc(p) >= UMBRAL_MISMO]
    if len(mismo) >= MIN_MISMO:
        return mismo, "mismo_problema"
    if u:
        return u[:TOPE_PARECIDOS], "parecidos"
    return [], "sin_base"


def wilson(k: float, n: float, z: float = 1.96) -> tuple[float, float] | None:
    if n <= 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(max(0.0, c - m), 3), round(min(1.0, c + m), 3)


def _cuenta(base: list, campo: str) -> Counter:
    return Counter(_pl(p).get(campo) for p in base if _pl(p).get(campo))


def _dist(cnt: Counter) -> dict:
    n = sum(cnt.values())
    return {k: round(v / n, 3) for k, v in cnt.most_common()} if n else {}


def tipo_de_base(base: list, pedido: str | None) -> str | None:
    if pedido:
        return pedido
    c = _cuenta(base, "j_tipo")
    return c.most_common(1)[0][0] if c else None


def agregar(local: list, nacional: list | None = None, tipo: str | None = None,
            circuito: str | None = None) -> dict:
    """La `estadistica` de /api/jurimetria. `local`: puntos del circuito pedido (o los nacionales si no hubo
    circuito); `nacional`: los mismos sin filtro de circuito, para prestar base y para el reparto por circuito."""
    nacional = local if nacional is None else nacional
    base_l, clase_l = base_de(local)
    base_n, clase_n = base_de(nacional)
    t = tipo_de_base(base_l or base_n, tipo)
    if t:
        base_l = [p for p in base_l if _pl(p).get("j_tipo") == t]
        base_n = [p for p in base_n if _pl(p).get("j_tipo") == t]
    pro_l, pro_n = _cuenta(base_l, "j_prospera"), _cuenta(base_n, "j_prospera")
    n_l, n_n = sum(pro_l.values()), sum(pro_n.values())
    p_n = {k: v / n_n for k, v in pro_n.items()} if n_n else {}
    cats = ("si", "parcial", "no", "sin_fondo")
    # Sin una sola sentencia propia no hay predicción del circuito: lo nacional va aparte, en `nacional`, y no se
    # pinta como si fuera de aquí.
    if circuito and 0 < n_l < 40 and n_n:
        adj = {k: (pro_l.get(k, 0) + PESO_NACIONAL * p_n.get(k, 0)) / (n_l + PESO_NACIONAL) for k in cats}
        prestado = True
    else:
        adj = {k: pro_l.get(k, 0) / n_l for k in cats} if n_l else {}
        prestado = False
    adj = {k: round(v, 3) for k, v in adj.items() if v > 0}

    # Lo que la pantalla ya sabe pintar.
    legado = Counter()
    for k, v in adj.items():
        legado[LEGADO[k]] += v
    probabilidades = {k: round(v, 3) for k, v in legado.most_common()}
    if probabilidades and n_l:
        sentido_probable = max(("CONCEDE", "NIEGA", "SOBRESEE"), key=lambda k: probabilidades.get(k, 0))
    else:
        sentido_probable = SIN_BASE

    # Intervalo del lado que se propone, con la base PROPIA (lo prestado no cuenta como evidencia).
    lado = {"CONCEDE": ("si", "parcial"), "NIEGA": ("no",), "SOBRESEE": ("sin_fondo",)}.get(sentido_probable, ())
    k_lado = sum(pro_l.get(x, 0) for x in lado)
    intervalo = wilson(k_lado, n_l) if n_l else None
    ancho = (intervalo[1] - intervalo[0]) if intervalo else 1.0
    if n_l >= 30 and ancho <= 0.30:
        confianza = "alta"
    elif n_l >= 12:
        confianza = "media"
    else:
        confianza = "baja"

    por_circuito = {}
    agrupado = defaultdict(Counter)
    for p in base_n:
        pl = _pl(p)
        if pl.get("j_prospera") and pl.get("circuito"):
            agrupado[str(pl["circuito"])][LEGADO[pl["j_prospera"]]] += 1
    for c, cnt in agrupado.items():
        por_circuito[c] = _dist(cnt)

    por_periodo = {}
    for per in ("anterior", "nueva_integracion"):
        sub = Counter(_pl(p).get("j_prospera") for p in base_l
                      if _pl(p).get("j_periodo") == per and _pl(p).get("j_prospera"))
        if sum(sub.values()):
            por_periodo[per] = {"n": sum(sub.values()), "prospera": _dist(sub)}

    ponentes = Counter()
    factores = Counter()
    ratio_s, ratio_n = 0.0, 0
    for p in base_l:
        pl = _pl(p)
        m = pl.get("j_ponente") or (str(pl.get("magistrado_ponente") or "").strip() or None)
        if m and m not in ("null", "None"):
            ponentes[m] += 1
        f = str(pl.get("factor_determinante_sentido") or "").strip()
        if f and f not in ("null", "None"):
            factores[f] += 1
        r = pl.get("ratio_inoperantes")
        if r is not None:
            try:
                ratio_s += float(r)
                ratio_n += 1
            except (TypeError, ValueError):
                pass

    if clase_l == "sin_base" or not n_l:
        aviso = ("No hay sentencias del acervo suficientes para estos filtros"
                 + (f" en el circuito {circuito}" if circuito else "") + ".")
        if circuito and n_n:
            aviso += f" En el país hay {n_n} sobre el tema (ver «nacional»)."
    elif clase_l == "parecidos":
        aviso = (f"Ninguna sentencia resolvió exactamente este problema (similitud {int(UMBRAL_MISMO * 100)}% o "
                 f"más); la tendencia sale de las {n_l} más parecidas.")
    else:
        aviso = f"{n_l} sentencias resolvieron este mismo problema."
    if prestado and n_l:
        aviso += f" La base del circuito es corta ({n_l}) y se completa con la tendencia nacional."

    return {
        "sentido_probable": sentido_probable,
        "probabilidades": probabilidades,
        "n_base": n_l,
        "por_circuito": por_circuito,
        "factor_dominante": factores.most_common(1)[0][0] if factores else None,
        "magistrados_frecuentes": [m for m, _ in ponentes.most_common(5)],
        "ratio_inoperantes_historico": round(ratio_s / ratio_n, 3) if ratio_n else None,
        "confianza": confianza,
        # ── lo nuevo (22-nov-2026) ──
        "pregunta": "¿Prospera lo que pide quien promueve?",
        "tipo_asunto": t,
        "tipo_asunto_nombre": jc.NOMBRE_TIPO.get(t) if t else None,
        "base": clase_l,
        "prospera": adj,
        "prospera_conteo": dict(pro_l),
        "intervalo_95": intervalo,
        "prestado_de_nacional": prestado,
        "nacional": {"n": n_n, "prospera": _dist(pro_n), "base": clase_n},
        "quejoso": _dist(_cuenta(base_l, "j_quejoso")),
        "quejoso_n": sum(_cuenta(base_l, "j_quejoso").values()),
        "por_periodo": por_periodo,
        "por_tipo": {k: v for k, v in _cuenta(base_de(local)[0], "j_tipo").most_common()},
        "aviso": aviso,
    }


def frase(est: dict) -> str:
    """Una línea honesta para el prompt de la narrativa y para quien lea el JSON."""
    if est.get("sentido_probable") == SIN_BASE:
        return est.get("aviso") or "Sin base suficiente."
    t = est.get("tipo_asunto_nombre") or "este tipo de asunto"
    pr = est.get("prospera") or {}
    partes = [f"{NOMBRE_PROSPERA[k]} {round(v * 100)}%" for k, v in sorted(pr.items(), key=lambda kv: -kv[1])]
    iv = est.get("intervalo_95")
    ivs = f" (intervalo de 95%: {round(iv[0] * 100)}–{round(iv[1] * 100)}%)" if iv else ""
    return f"En {t}, lo que pide quien promueve: " + ", ".join(partes) + ivs + f". {est.get('aviso', '')}"
