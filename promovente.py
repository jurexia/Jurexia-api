"""QUIÉN ES LA PARTE Y QUIÉN SÓLO LA REPRESENTA.

POR QUÉ EXISTE. ADC 93/2026, v5 (22-sep-2026). David, en ponencia:

    «La persona física (Alondra Zúñiga Gutiérrez) no resiente el perjuicio ni
     tiene legitimación activa ad causam; quien resiente la afectación y es la
     quejosa directa es la persona moral (GDE Trading Company, S.A. de C.V.).
     La persona física únicamente ostenta la legitimación procesal activa
     (representación o personería) en términos de los artículos 6 y 11 de la
     Ley de Amparo. (…) Lo técnicamente pulcro es amparar a la persona
     jurídica: "ampara y protege a GDE Trading Company, S.A. de C.V., por
     conducto de su representante legal…".»

LA CAUSA. La ficha del encargo traía en `quejoso` la frase entera tal como la
leyó el auto de admisión —«Alondra Zúñiga Gutiérrez, representante legal de
GDE Trading Company, sociedad anónima de capital variable»— y el compositor la
copiaba en los doce sitios donde va el nombre: la carátula, la legitimación
(«quien está legitimada… por resentir el perjuicio») y el resolutivo («ampara
y protege a Alondra…»). `datos["representante"]` existía en el compositor y
nadie lo llenaba.

Aquí se separa UNA vez, en la fuente, y de ahí sale para todos: la parte es
la representada; el representante es quien comparece por ella. Sin modelo:
son cuatro fórmulas que el foro escribe siempre igual.
"""
from __future__ import annotations

import re

_FIGURA = (r"representante\s+legal|apoderad[oa](?:\s+(?:legal|general|especial))?"
           r"(?:\s+para\s+pleitos\s+y\s+cobranzas)?|administrador[a]?\s+[úu]nic[oa]|"
           r"administrador[a]?\s+general|gerente\s+general|director[a]?\s+general|"
           r"presidente\s+del\s+consejo\s+de\s+administraci[óo]n|mandatari[oa]|"
           r"albacea|tutor[a]?|autorizad[oa]|delegad[oa]|s[íi]ndic[oa]|"
           r"representante(?:\s+com[úu]n)?")

# «X, representante legal de Y» · «X, en su carácter de apoderada de Y» ·
# «X, como administrador único de Y» · «X, quien se ostenta como…»
_RX_REP_DE = re.compile(
    rf"^(?P<rep>.+?)[,;]?\s+(?:(?:en\s+su\s+car[áa]cter\s+de|en\s+su\s+calidad\s+de|"
    rf"como|quien\s+(?:se\s+ostenta|comparece|actúa|actua)\s+como|ostent[áa]ndose\s+como)\s+)?"
    rf"(?P<fig>{_FIGURA})\s+(?:de|del|de\s+la|de\s+los|de\s+las)\s+(?P<parte>.+?)\s*\.?$",
    re.I)
# «X, en representación de Y» · «X, en nombre y representación de Y»
_RX_EN_REP = re.compile(
    r"^(?P<rep>.+?)[,;]?\s+en\s+(?:nombre\s+y\s+)?representaci[óo]n\s+(?:de|del|de\s+la)\s+"
    r"(?P<parte>.+?)\s*\.?$", re.I)
# «Y, por conducto de su representante legal X» · «Y, a través de su apoderado X»
# · «Y, representada por X»
_RX_POR_CONDUCTO = re.compile(
    rf"^(?P<parte>.+?)[,;]?\s+(?:por\s+conducto\s+de|a\s+trav[ée]s\s+de|representad[oa]\s+por)\s+"
    rf"(?:su\s+|de\s+su\s+)?(?:(?P<fig>{_FIGURA})[,]?\s+)?(?P<rep>.+?)\s*\.?$", re.I)

# Señales de persona moral: bastan para saber que la parte no es una persona
# física, y con eso el género gramatical de «legitimada» / «quejosa».
_RX_MORAL = re.compile(
    r"\b(?:s\.?\s*a\.?(?:\s*de\s*c\.?\s*v\.?)?|s\.?\s*de\s*r\.?\s*l\.?|s\.?\s*c\.?|a\.?\s*c\.?|"
    r"sociedad\s+an[óo]nima|sociedad\s+civil|asociaci[óo]n\s+civil|sociedad\s+de\s+"
    r"responsabilidad\s+limitada|s\.?\s*a\.?\s*p\.?\s*i\.?|instituci[óo]n|banco|"
    r"ayuntamiento|municipio|secretar[íi]a|instituto|comisi[óo]n|universidad|"
    r"sindicato|cooperativa|fideicomiso|empresa|compa[ñn][íi]a|company|corporation|"
    r"inc\.?|llc|ltd\.?|s\.?\s*a\.?\s*s\.?)\b", re.I)


def _limpia(x: str) -> str:
    t = " ".join((x or "").replace("\n", " ").split()).strip(" ,;")
    # EL PUNTO DE UNA ABREVIATURA SE QUEDA: «S.A. DE C.V.» no es una frase
    # que termina; sólo se quita el punto final si cierra una palabra entera.
    if t.endswith(".") and not re.search(r"\b[A-Za-zÁÉÍÓÚÑ]\.$", t):
        t = t[:-1].rstrip(" ,;")
    return t


def separar(texto: str) -> dict:
    """{parte, representante, figura, moral} a partir de la frase de la ficha.

    Si no hay fórmula de representación, `parte` es el texto tal cual y
    `representante` va vacío: nada cambia para quien ya venía bien.
    """
    t = _limpia(texto)
    if not t:
        return {"parte": "", "representante": "", "figura": "", "moral": False}
    for rx, orden in ((_RX_REP_DE, "rep-de"), (_RX_EN_REP, "en-rep"),
                      (_RX_POR_CONDUCTO, "por-conducto")):
        m = rx.match(t)
        if not m:
            continue
        parte = _limpia(m.group("parte"))
        rep = _limpia(m.group("rep"))
        fig = _limpia(m.groupdict().get("fig") or "")
        # UNA FÓRMULA MAL LEÍDA ES PEOR QUE NINGUNA: las dos mitades tienen que
        # parecer nombres (dos palabras al menos) y no ser la misma.
        if len(parte.split()) < 2 or len(rep.split()) < 2 or parte.lower() == rep.lower():
            continue
        return {"parte": parte, "representante": rep,
                "figura": (fig or "representante").lower(),
                "moral": bool(_RX_MORAL.search(parte))}
    return {"parte": t, "representante": "", "figura": "", "moral": bool(_RX_MORAL.search(t))}


def es_moral(nombre: str) -> bool:
    return bool(_RX_MORAL.search(nombre or ""))


def por_conducto(parte: str, representante: str, figura: str = "") -> str:
    """«GDE Trading Company, S.A. de C.V., por conducto de su representante
    legal Alondra Zúñiga Gutiérrez» — la fórmula para el resolutivo cuando se
    quiere nombrar a quien compareció; sin representante, sólo la parte."""
    p, r = _limpia(parte), _limpia(representante)
    if not r:
        return p
    f = (figura or "representante legal").strip().lower()
    return f"{p}, por conducto de su {f} {r}"
