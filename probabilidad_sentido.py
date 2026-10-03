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


# LOS TIPOS QUE EL TALLER NO PROYECTA PERO EL TRIBUNAL SÍ DECIDE (2-oct-2026).
# `tipos_asunto.normalizar` sólo conoce los cuatro del taller (la reclamación y
# el impedimento quedaron fuera por decisión de David, ver su cabecera); la
# tasa de estos tres está medida igual y no se pierde por la grafía.
_ALIAS_TIPO = {
    "reclamacion": "reclamacion", "recurso_de_reclamacion": "reclamacion",
    "recurso_reclamacion": "reclamacion", "rec": "reclamacion",
    "inconformidad": "inconformidad", "recurso_de_inconformidad": "inconformidad",
    "recurso_inconformidad": "inconformidad", "inc": "inconformidad",
    "impedimento": "impedimento", "imp": "impedimento",
}


def clave_tipo(tipo) -> str:
    """La clave con que se busca la tasa: la de `tipos_asunto.normalizar` para
    los cuatro del taller, la propia para reclamación, inconformidad e
    impedimento, y la grafía limpia si no se reconoce."""
    import unicodedata
    x = unicodedata.normalize("NFKD", str(tipo or "").strip().lower())
    x = "".join(c for c in x if not unicodedata.combining(c))
    x = x.replace(" ", "_").replace("-", "_")
    if x in _ALIAS_TIPO:
        return _ALIAS_TIPO[x]
    try:
        import tipos_asunto as _ta
        return _ta.normalizar(x) or x
    except Exception:                                   # pragma: no cover
        return x


def tasa(tipo: str, cal: dict | None = None) -> tuple:
    """(probabilidad de que prospere, asuntos que la sostienen) del tipo; (None, 0)
    si el tipo no está medido."""
    cal = cargar() if cal is None else cal
    t = (cal.get("tasas") or {}).get(clave_tipo(tipo)) or {}
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
             cal: dict | None = None, propio: bool = True) -> dict:
    """La probabilidad de que el asunto PROSPERE y el lado que se propone.

    {p_prospera, lado: "prospera"|"no_prospera"|None, tasa, n_tasa, precedentes,
     motor, explicacion}. `lado` None sólo si no hay tasa para el tipo ni voto del
    motor: entonces no hay número que dar."""
    cal = cargar() if cal is None else cal
    tipo = clave_tipo(tipo)
    p0, n0 = tasa(tipo, cal)
    ev = evidencia(filas or [], cal)
    v = voto_motor(sentido_motor)
    if p0 is None:
        # Sin tasa medida para el tipo no se inventa: decide el motor, como antes.
        lado = None if v is None else ("prospera" if v else "no_prospera")
        return {"p_prospera": None, "lado": lado, "tasa": None, "n_tasa": 0,
                "precedentes": ev, "motor": v, "fuente": "motor",
                "explicacion": ("Sin tasa medida para este tipo de asunto: manda la propuesta del motor."
                                if v is not None else
                                "Sin tasa medida para este tipo de asunto y sin propuesta del motor: "
                                "no hay número que dar.")}
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
            "explicacion": explicar(tipo, p, p0, n0, ev, v, lado, propio)}


# En plural, porque se lee «la tasa del tribunal en sus amparos directos».
_NOMBRE_TIPO = {"amparo_directo": "amparos directos", "amparo_revision": "amparos en revisión",
                "queja": "quejas", "revision_fiscal": "revisiones fiscales",
                "reclamacion": "recursos de reclamación", "inconformidad": "recursos de inconformidad",
                "impedimento": "impedimentos"}
_PROSPERA_TXT = {"amparo_directo": ("conceder", "negar"),
                 "amparo_revision": ("que el recurso prospere", "que no prospere"),
                 "queja": ("que la queja sea fundada", "que sea infundada"),
                 "revision_fiscal": ("que el recurso prospere", "que no prospere"),
                 "reclamacion": ("que la reclamación sea fundada", "que sea infundada"),
                 "inconformidad": ("que la inconformidad sea fundada", "que sea infundada"),
                 "impedimento": ("que el impedimento sea fundado", "que sea infundado")}


def _pct(x: float) -> str:
    return f"{int(round(x * 100))}%"


def explicar(tipo, p, p0, n0, ev, v, lado, propio: bool = True) -> str:
    """Una frase para el secretario: el número y de dónde sale.

    (2-oct-2026) «Sale de … ningún precedente» no se lee: lo que no hay se
    dice aparte. `propio` False: el tribunal que proyecta no es el de la tasa
    y se dice que es una referencia, no su estadística."""
    tipo = clave_tipo(tipo)
    si, no = _PROSPERA_TXT.get(tipo, ("que prospere", "que no prospere"))
    gana = si if lado == "prospera" else no
    prob = p if lado == "prospera" else 1 - p
    nombre = _NOMBRE_TIPO.get(tipo, "asuntos de " + str(tipo or "").replace("_", " "))
    if propio:
        partes = [f"la tasa del tribunal en sus {nombre} (prospera el {_pct(p0)} de {n0:,} asuntos)"]
    else:
        partes = [f"la tasa de referencia del Tercer Tribunal Colegiado del Vigésimo Segundo "
                  f"Circuito en sus {nombre} (prospera el {_pct(p0)} de {n0:,} asuntos)"]
    fav = sum(1 for f in ev["filas"] if f["prospero"]) if ev.get("n") else 0
    if ev.get("n"):
        partes.append(f"{ev['n']} precedente{'s' if ev['n'] != 1 else ''} del tribunal sobre el mismo "
                      f"problema ({fav} en que prosperó y {ev['n'] - fav} en que no)")
    if v is not None:
        partes.append(f"la lectura del motor, que {'apuntaba a que prospera' if v else 'apuntaba a que no prospera'} "
                      f"(su voto pesa poco: así lo midió el banco de 21 asuntos)")
    frase = (f"Se propone {gana}, con una probabilidad del {_pct(prob)}. Sale de "
             + (", ".join(partes[:-1]) + " y " if len(partes) > 1 else "") + partes[-1] + ".")
    if not ev.get("n"):
        frase += " No hay precedentes del tribunal sobre el mismo problema."
    return frase


# ═══════════════════════════════════════════════════════════════════════════
# LA CAPA DE PROBABILIDAD SOBRE LA PROPUESTA (2-oct-2026)
#
# Corre en `main._taller_proponer_nucleo` detrás de la bandera
# «propuesta_por_probabilidad», después de emparejar y de lo vinculado por la
# ejecutoria y ANTES de `desenlace.reconciliar` y del árbol. Aquí vive la parte
# pura —qué lado, si se voltea, cómo se intercambian las vías— para probarla
# sin red; main pone las lecturas (precedentes, la razón que falte) alrededor.
#
# EL VOLTEO NO ESCRIBE RAZONES NUEVAS SI YA LAS HAY. El motor escribe, en la
# misma llamada, la vía contraria «como si la defendiera» (`alternativa`, con
# sentido, razón, efecto y apoyos) y la suerte de cada accesorio en las dos
# vías (`checklist`: `con_propuesta` / `con_alternativa`). Si gana el otro
# lado, se intercambian: la global pasa a ser la alternativa y lo que propuso
# el motor queda como alternativa, con su razón como objeción (`en_contra`).
# Sólo si no hay alternativa escrita main pide UNA razón (`_f5.razonar`).
# ═══════════════════════════════════════════════════════════════════════════

def confianza_de(p_lado) -> str:
    """La confianza que se enseña, de la probabilidad del lado propuesto: alta
    desde 0.80, media desde 0.65, baja por debajo."""
    try:
        x = float(p_lado)
    except (TypeError, ValueError):
        return ""
    return "alta" if x >= 0.80 else "media" if x >= 0.65 else "baja"


def _g(x, k, d=None):
    return x.get(k, d) if isinstance(x, dict) else getattr(x, k, d)


def _s(x, k, v) -> None:
    if isinstance(x, dict):
        x[k] = v
    else:
        setattr(x, k, v)


def _norm(t) -> str:
    import unicodedata
    t = unicodedata.normalize("NFKD", str(t or "").lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join("".join(c if c.isalnum() else " " for c in t).split())[:400]


def indice_principal(problemas: list, propuestas: list, jerarquias: dict | None = None) -> int:
    """El índice del problema principal entre `propuestas` (alineadas con los
    problemas): el de jerarquía «principal» en la fase 3 o en `jerarquias`
    ({pregunta: jerarquia}); si ninguno lo es, el primero. -1 sin problemas."""
    n = len(propuestas or [])
    if not n:
        return -1
    for i, p in enumerate(problemas or []):
        if i < n and isinstance(p, dict) and str(p.get("jerarquia") or "").strip().lower() == "principal":
            return i
    jer = {_norm(k): str(v or "").strip().lower() for k, v in (jerarquias or {}).items()}
    for i, pr in enumerate(propuestas):
        if jer.get(_norm(_g(pr, "problema", ""))) == "principal":
            return i
    return 0


def _sentido_de_lado(lado: str) -> str:
    return "fundado" if lado == "prospera" else "infundado"


def _intercambiar_checklist(checklist: list) -> None:
    """La suerte «con la propuesta» y «con la alternativa» se escriben referidas
    a la vía del motor (`arbol_decision._suerte`): si la propuesta pasa a ser
    la alternativa, se intercambian para que sigan diciendo lo mismo."""
    for c in checklist or []:
        if isinstance(c, dict) and ("con_propuesta" in c or "con_alternativa" in c):
            c["con_propuesta"], c["con_alternativa"] = c.get("con_alternativa", ""), c.get("con_propuesta", "")


def aplicar(glob, propuestas: list, problemas: list, tipo: str, filas: list | None = None,
            jerarquias: dict | None = None, cal: dict | None = None, propio: bool = True) -> dict:
    """Fija EN SITIO el lado de la propuesta con la probabilidad.

    `glob`: la `Global` de la fase 5 (o un dict con sus campos); `propuestas`:
    las `Propuesta` ya emparejadas con `problemas` (las huérfanas sin
    sentido). `filas`: los precedentes del propio tribunal para el principal.

    Devuelve {probabilidad, principal, volteada, necesita_razon, sentido,
    avisos}. `probabilidad` es la de `calcular` más `volteada` y
    `sentido_motor`. `necesita_razon`: el lado ganador no tiene razón escrita
    —el motor no escribió la vía contraria, o no propuso nada— y main la pide.
    Nunca lanza por datos raros: lo que no entiende, lo deja como estaba."""
    avisos: list = []
    i = indice_principal(problemas, propuestas, jerarquias)
    pral = propuestas[i] if i >= 0 else None
    s_glob = str(_g(glob, "sentido", "") or "").strip().lower()
    s_pral = str(_g(pral, "sentido", "") or "").strip().lower() if pral is not None else ""
    s_motor = s_glob or s_pral
    prob = calcular(tipo, filas, s_motor or None, cal, propio=propio)
    prob["sentido_motor"] = s_motor
    prob["volteada"] = False
    info = {"probabilidad": prob, "principal": i, "volteada": False,
            "necesita_razon": False, "sentido": s_glob, "avisos": avisos}
    lado = prob.get("lado")
    if lado not in ("prospera", "no_prospera"):
        return info
    quiere = lado == "prospera"
    v = voto_motor(s_motor) if s_motor else None

    # LA CONFIANZA QUE SE ENSEÑA SALE DEL NÚMERO; la del modelo se guarda.
    _p = prob.get("p_prospera")
    if _p is not None:
        _s(glob, "confianza_motor", str(_g(glob, "confianza", "") or ""))
        _s(glob, "confianza", confianza_de(_p if quiere else 1 - _p))

    if v is not None and bool(v) == quiere:
        # EL MOTOR YA ESTÁ DEL LADO QUE GANA. Si no escribió la global (sólo
        # el problema principal), la del asunto es la del principal: es de la
        # que cuelga.
        if not s_glob and pral is not None and s_pral:
            _s(glob, "sentido", s_pral)
            _s(glob, "razon", str(_g(pral, "razon", "") or ""))
            _s(glob, "apoyos", list(_g(pral, "apoyos", None) or []))
            if not str(_g(glob, "problema_que_decide", "") or "").strip():
                _s(glob, "problema_que_decide", str(_g(pral, "problema", "") or ""))
            _s(glob, "alcanza", True)
            avisos.append("El motor no escribió la propuesta del asunto entero: se toma la del "
                          "problema principal, que es del que cuelga el resultado.")
        elif pral is not None and not s_pral and s_glob:
            _s(pral, "sentido", s_glob)
            _s(pral, "razon", str(_g(glob, "razon", "") or ""))
            _s(pral, "apoyos", list(_g(glob, "apoyos", None) or []))
            _s(pral, "alcanza", True)
            _s(pral, "origen", "motor")
        info["sentido"] = str(_g(glob, "sentido", "") or "")
        _s(glob, "alcanza", bool(info["sentido"]))
        return info

    # ── EL OTRO LADO GANA (o el motor no dijo nada) ──
    alt = _g(glob, "alternativa", None)
    alt = alt if isinstance(alt, dict) else {}
    s_alt = str(alt.get("sentido") or "").strip().lower()
    _vieja = {"sentido": s_glob, "razon": str(_g(glob, "razon", "") or ""),
              "efecto": str(_g(glob, "efecto", "") or ""),
              "apoyos": list(_g(glob, "apoyos", None) or []),
              "sostenida": bool(_g(glob, "sostenida", True))}
    if s_alt and voto_motor(s_alt) is not None and bool(voto_motor(s_alt)) == quiere \
            and str(alt.get("razon") or "").strip():
        nuevo = {"sentido": s_alt, "razon": str(alt.get("razon") or ""),
                 "efecto": str(alt.get("efecto") or ""), "apoyos": list(alt.get("apoyos") or [])}
    else:
        nuevo = {"sentido": _sentido_de_lado(lado), "razon": "", "efecto": "", "apoyos": []}
        info["necesita_razon"] = True
    _s(glob, "sentido", nuevo["sentido"])
    _s(glob, "razon", nuevo["razon"])
    _s(glob, "efecto", nuevo["efecto"])
    _s(glob, "apoyos", nuevo["apoyos"])
    _s(glob, "alcanza", True)
    _s(glob, "sostenida", bool(nuevo["apoyos"]))
    if s_glob:
        # Lo que propuso el motor queda como la vía contraria, entera, y su
        # razón es la mejor objeción a la que se propone.
        _s(glob, "alternativa", {k: _vieja[k] for k in ("sentido", "razon", "efecto", "apoyos")})
        _s(glob, "en_contra", _vieja["razon"][:400])
        _intercambiar_checklist(_g(glob, "checklist", None) or [])
    if not str(_g(glob, "problema_que_decide", "") or "").strip() and pral is not None:
        _s(glob, "problema_que_decide", str(_g(pral, "problema", "") or ""))
    volteada = v is not None
    prob["volteada"] = volteada
    info["volteada"] = volteada
    info["sentido"] = nuevo["sentido"]

    # EL PRINCIPAL VA CON EL ASUNTO: toma el sentido y la razón del lado que
    # gana si iba por el otro o no tenía. Lo que el motor le propuso queda en
    # `sentido_motor`/`razon_motor` —NO en `sentido_propio`, que el árbol lee
    # como «la vía del motor» y, al resolver, tumbaría los accesorios—.
    if pral is not None and (not s_pral or voto_motor(s_pral) is None
                             or bool(voto_motor(s_pral)) != quiere):
        if s_pral:
            _s(pral, "sentido_motor", s_pral)
            _s(pral, "razon_motor", str(_g(pral, "razon", "") or ""))
        _s(pral, "sentido", nuevo["sentido"])
        _s(pral, "razon", nuevo["razon"])
        _s(pral, "apoyos", list(nuevo["apoyos"]))
        _s(pral, "alcanza", True)
        _s(pral, "sostenida", bool(nuevo["apoyos"]))
        _s(pral, "origen", "probabilidad")
    if volteada:
        prob["explicacion"] = (prob.get("explicacion", "") + f" El motor se inclinaba por lo contrario "
                               f"(«{s_motor.replace('_', ' ')}»): su razón queda como vía contraria.")
        avisos.append("LA PROPUESTA SE VOLTEÓ POR PROBABILIDAD: " + prob["explicacion"])
    elif not s_motor:
        avisos.append("El motor no propuso ningún sentido; se propone el más probable: "
                      + prob.get("explicacion", ""))
    return info
