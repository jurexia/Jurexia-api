"""LAS CONSTANCIAS QUE HACEN FALTA VER PARA RESOLVER — y que sólo el
secretario tiene.

POR QUÉ EXISTE. David, 22-sep-2026: «elaborar una sentencia de tribunal
terminal con sólo confrontar la sentencia reclamada y lo alegado por el
quejoso puede llevar al modelo a una resolución sesgada por la falta de
información que sólo un secretario puede verificar en las constancias del
juicio de origen. Es vital que el modelo detecte cuándo resulte
estrictamente indispensable, para dar solución, información sobre alguna
constancia —ya sea que la detalle en un cuadro de texto o que adjunte el
documento faltante—».

Lo que había: la fase 3 marcaba en `apoyo.motivo = "constancia"` que un
planteamiento se apoya en algo que consta; la fase 5 decía «no alcanza» y en
prosa qué le faltaba; y la ventana de contexto recibía cualquier cosa sin
saber a qué respondía. Tres puertas que no se hablaban.

Ahora la propuesta DECLARA, en un campo, qué constancias son indispensables
y para qué; la pantalla las pide una por una (texto o documento); cada
aporte viaja rotulado —«[CONSTANCIA · el acuerdo de 13 de agosto]»— para que
se sepa cuál se contestó; y el estudio recibe la lista de las que NO se
aportaron con la orden de no suponer su contenido.
"""
from __future__ import annotations

import re

ROTULO = "CONSTANCIA"
_RX_ROTULO = re.compile(r"\[\s*CONSTANCIA\s*[·:\-–—]\s*([^\]]{3,200})\]", re.I)
MAX = 6


def _limpia(x) -> str:
    return " ".join(str(x or "").split()).strip(" .;")


def _norm(x: str) -> str:
    import unicodedata
    t = unicodedata.normalize("NFKD", _limpia(x).lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9 ]+", " ", t)


def _parecidas(a: str, b: str) -> bool:
    pa, pb = set(_norm(a).split()), set(_norm(b).split())
    pa = {w for w in pa if len(w) > 3}
    pb = {w for w in pb if len(w) > 3}
    if not pa or not pb:
        return False
    inter = len(pa & pb)
    return inter >= 3 or inter / max(1, min(len(pa), len(pb))) >= 0.6


def normalizar(lista, problemas: list = None) -> list:
    """[{que, para_que, indispensable, problema}] limpia, sin repetidas, con
    las indispensables primero y como mucho MAX."""
    fuera: list = []
    for c in (lista or []):
        if isinstance(c, str):
            c = {"que": c}
        if not isinstance(c, dict):
            continue
        que = _limpia(c.get("que") or c.get("constancia") or c.get("documento"))
        if len(que) < 6:
            continue
        item = {"que": que[:200],
                "para_que": _limpia(c.get("para_que") or c.get("razon") or c.get("explicacion"))[:300],
                "indispensable": bool(c.get("indispensable", True)),
                "problema": int(c.get("problema") or 0) if str(c.get("problema") or "").strip().isdigit() else 0}
        if any(_parecidas(item["que"], x["que"]) for x in fuera):
            for x in fuera:
                if _parecidas(item["que"], x["que"]):
                    x["indispensable"] = x["indispensable"] or item["indispensable"]
                    if not x["para_que"]:
                        x["para_que"] = item["para_que"]
            continue
        fuera.append(item)
    fuera.sort(key=lambda x: (0 if x["indispensable"] else 1))
    return fuera[:MAX]


def de_fase3(problemas: list) -> list:
    """Lo que la fase 3 ya marcaba: un apoyo que descansa en una constancia."""
    fuera = []
    for i, p in enumerate(problemas or [], 1):
        if not isinstance(p, dict):
            continue
        ap = p.get("apoyo") if isinstance(p.get("apoyo"), dict) else {}
        if str(ap.get("motivo") or "").strip().lower() == "constancia" and ap.get("explicacion"):
            fuera.append({"que": str(ap["explicacion"])[:200],
                          "para_que": f"sostiene el planteamiento {i}",
                          "indispensable": False, "problema": i})
    return fuera


def rotular(etiqueta: str, texto: str) -> str:
    """El aporte, con su rótulo delante, para que se sepa a qué responde."""
    e = _limpia(etiqueta)
    t = (texto or "").strip()
    if not e:
        return t
    return f"[{ROTULO} · {e}]\n{t}"


def aportadas_en(contexto: str) -> list:
    """Los rótulos de constancia que viajan en el contexto."""
    return [_limpia(m.group(1)) for m in _RX_ROTULO.finditer(contexto or "")]


def faltantes(pedidas: list, contexto: str) -> list:
    """Las pedidas que ningún aporte rotulado cubre."""
    hay = aportadas_en(contexto)
    fuera = []
    for c in normalizar(pedidas):
        if not any(_parecidas(c["que"], h) for h in hay):
            fuera.append(c)
    return fuera


def bloque_para_estudio(pedidas: list, contexto: str) -> str:
    """Lo que el estudio tiene que saber: qué se pidió ver y no llegó."""
    falt = faltantes(pedidas, contexto)
    apor = [c for c in normalizar(pedidas) if c not in falt]
    if not falt and not apor:
        return ""
    partes = ["", "═" * 71, "CONSTANCIAS DEL JUICIO DE ORIGEN", "═" * 71]
    if apor:
        partes.append("Las que se pidieron y SÍ están arriba, rotuladas «[CONSTANCIA · …]»: "
                      + "; ".join(c["que"] for c in apor)
                      + ". Cítalas como constancias de autos y resuelve con ellas.")
    if falt:
        partes.append("LAS QUE HACÍAN FALTA Y NO SE APORTARON:")
        for c in falt:
            partes.append(f"  · {c['que']}" + (f" — {c['para_que']}" if c["para_que"] else "")
                          + (" [INDISPENSABLE]" if c["indispensable"] else ""))
        partes.append(
            "NO SUPONGAS SU CONTENIDO. Lo que dependa de una de ellas se dice como "
            "NO ACREDITADO EN EL MATERIAL y se resuelve con la carga de la prueba y "
            "con lo que sí consta; si sin ella el planteamiento no se puede "
            "decidir, dilo en el apartado ADVERTENCIAS con su nombre, para que "
            "quien firma la busque en el expediente antes de listar.")
    return "\n".join(partes) + "\n"


def aviso_faltantes(pedidas: list, contexto: str, al_reves: bool = False) -> str:
    """El aviso de las indispensables que no llegaron.

    `al_reves` = el secretario resolvió en dirección contraria a la propuesta
    del motor (23-sep-2026, revisión 711/2025). Entonces estas constancias las
    pedía el razonamiento DEL MOTOR, y decir «lo que dependía de ellas va como
    no acreditado» y «la solución puede cambiar» es socavar una decisión que
    ya está tomada por quien firma. Se avisa de otra manera: que el motor las
    quería para su vía, por si el secretario quiere tenerlas a la vista."""
    falt = [c for c in faltantes(pedidas, contexto) if c["indispensable"]]
    if not falt:
        return ""
    lista = "; ".join(f"«{c['que']}»" + (f" ({c['para_que']})" if c["para_que"] else "")
                      for c in falt)
    if al_reves:
        return ("CONSTANCIAS QUE EL MOTOR HABRÍA QUERIDO VER para la vía que él "
                "proponía y que no se tomó: " + lista + ". El proyecto se resolvió "
                "por el criterio del secretario y no depende de ellas; se listan "
                "por si quien firma quiere tenerlas a la vista en el expediente.")
    return ("FALTAN CONSTANCIAS INDISPENSABLES: el proyecto se escribió sin ver "
            + lista
            + ". Lo que dependía de ellas va como no acreditado. Apórtalas en la "
              "pantalla de decisión —texto o documento— y vuelve a generar: la "
              "solución puede cambiar.")
