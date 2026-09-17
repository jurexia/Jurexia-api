"""EL DESENLACE LO DICTA LA TARJETA FINAL, Y SE DICE IGUAL EN TODO EL PROYECTO.

═══════════════════════════════════════════════════════════════════════════════
POR QUÉ EXISTE
═══════════════════════════════════════════════════════════════════════════════
Revisión fiscal 2/2026, 17-sep-2026. La tarjeta final del motor decía INFUNDADO:
«El recurso no debe prosperar (…) El tercer planteamiento es fundado sólo
respecto del alcance del antecedente invocado, pero resulta insuficiente».
El proyecto salió así, en tres renglones:

    estudio     «Por ello, se confirma la sentencia de siete de octubre…»
    cierre      «…al resultar en parte infundados y en parte fundados lo
                 planteado, lo procedente es revocar la sentencia recurrida.»
    resolutivos «PRIMERO. Se revoca la sentencia… SEGUNDO. Se ordena a la
                 SALA REGIONAL… dictar otra sentencia…»
    síntesis    «…no justifica revocarla.»

David: «ESTO ES GRAVÍSIMO Y NO SIGUE LA TARJETA FINAL DE CONGRUENCIA».

LA CAUSA. El desenlace se decidía en dos sitios con dos fuentes. El estudio y la
síntesis —que escribe el modelo— lo sacaban de la tarjeta final. El cierre y los
resolutivos —que se arman por regla— lo sacaban de las calificaciones con «basta
un fundado para que prospere». Y el tercer planteamiento viajó calificado
«fundado» a secas, cuando la propia tarjeta decía que era fundado PERO
INSUFICIENTE. `modos_decision.repartir` reconciliaba las calificaciones con el
resultado sólo en un sentido —la sustracción de materia, cuando el principal
prospera— y para el inverso no había regla.

Y NADIE LO VIO: la revisión de congruencia sólo conocía las fórmulas del amparo
—«ampara y protege» / «no ampara ni protege»—; en un recurso, «se confirma» en
el estudio contra «Se revoca» en el resolutivo pasaba limpio.

═══════════════════════════════════════════════════════════════════════════════
LA REGLA
═══════════════════════════════════════════════════════════════════════════════
Cuando hay tarjeta global —el secretario la dictó o es la del motor—, ELLA dice
si el asunto prospera, y las calificaciones por tema tienen que decir lo mismo:

  · Si NO prospera, un tema calificado con algo que prospera es, jurídicamente,
    FUNDADO PERO INSUFICIENTE: fundado y sin fuerza para cambiar el resultado.
  · Si prospera y ningún tema prospera, el principal lleva el sentido global.

En la vía POR PROBLEMA no hay tarjeta global: las marcas del secretario SON la
tarjeta y aquí no se tocan.

Se aplica en la fase que propone —para que la pantalla ya muestre la tarjeta
coherente— y otra vez al resolver, para lo que llegue por otro camino. Y el
documento terminado se comprueba: si una parte dice confirmar y otra revocar,
el proyecto no sale en silencio.
"""
from __future__ import annotations

import re

import tipos_asunto as _ta

INSUFICIENTE = "fundado_insuficiente"


def _get(x, campo, defecto=""):
    return x.get(campo, defecto) if isinstance(x, dict) else getattr(x, campo, defecto)


def _set(x, campo, valor):
    if isinstance(x, dict):
        x[campo] = valor
    else:
        setattr(x, campo, valor)


def prospera_la_tarjeta(sentido_global: str):
    """True / False si la tarjeta global lo dice; None si no hay tarjeta global."""
    s = (sentido_global or "").strip().lower()
    if not s:
        return None
    return bool(_ta.prospera(s))


def reconciliar(items: list, sentido_global: str, jerarquias: dict = None) -> list:
    """Pone las calificaciones de acuerdo con la tarjeta. Devuelve los avisos.

    `items` son Propuestas, Criterios o dicts con `problema` y `sentido`.
    `jerarquias` es {texto del problema: "principal"|"accesorio"} cuando el
    objeto no la trae.
    """
    va = prospera_la_tarjeta(sentido_global)
    if va is None or not items:
        return []
    jer = jerarquias or {}
    avisos = []

    def _es_principal(x, i):
        j = str(_get(x, "jerarquia", "") or jer.get(str(_get(x, "problema", "")), "")).lower()
        return j == "principal" if j else i == 0

    if va is False:
        for x in items:
            s = str(_get(x, "sentido", "") or "").strip().lower()
            if s and _ta.prospera(s):
                _set(x, "sentido", INSUFICIENTE)
                avisos.append(
                    f"«{str(_get(x, 'problema', ''))[:80]}» venía calificado "
                    f"{s.replace('_', ' ').upper()}, y la tarjeta final dice que el "
                    f"asunto NO prospera ({sentido_global.replace('_', ' ')}). Un "
                    f"planteamiento fundado que no cambia el resultado es FUNDADO "
                    f"PERO INSUFICIENTE, y así se califica: de otro modo el cierre y "
                    f"los resolutivos revocarían lo que el estudio confirma. Si "
                    f"quieres que prospere, cambia la calificación global.")
        return avisos

    # va is True: tiene que prosperar al menos uno
    if any(_ta.prospera(str(_get(x, "sentido", "") or "")) for x in items):
        return avisos
    principal = next((x for i, x in enumerate(items) if _es_principal(x, i)), items[0])
    antes = str(_get(principal, "sentido", "") or "sin calificar")
    _set(principal, "sentido", sentido_global.strip().lower())
    avisos.append(
        f"La tarjeta final dice que el asunto PROSPERA "
        f"({sentido_global.replace('_', ' ')}) y ningún planteamiento estaba "
        f"calificado así. El principal —«{str(_get(principal, 'problema', ''))[:80]}»— "
        f"pasa de {antes.replace('_', ' ')} a {sentido_global.replace('_', ' ')}, "
        f"para que el resolutivo diga lo mismo que la tarjeta.")
    return avisos


# ═══════════════════════════════════════════════════════════════════════════
# LA COMPROBACIÓN SOBRE EL DOCUMENTO TERMINADO
# ═══════════════════════════════════════════════════════════════════════════
# Para los recursos: el desenlace se dice con confirmar / revocar / modificar,
# y tiene que decirse igual en el estudio, en el cierre, en los resolutivos y en
# la síntesis. El amparo directo ya lo vigila `revisar_congruencia` con sus
# fórmulas propias.

_TIPOS_CON_SENTENCIA_RECURRIDA = {"revision_fiscal", "amparo_revision"}

# Lo que dice que PROSPERA el recurso.
_RX_REVOCA = re.compile(
    r"\bse\s+revoca\b|\bse\s+modifica\s+la\s+sentencia\b|"
    r"\blo\s+procedente\s+es\s+(?:revocar|modificar)\b|"
    r"\bprocede\s+(?:revocar|modificar)\b|"
    r"\bdebe\s+(?:revocarse|modificarse)\b|"
    r"\b(?:es|resulta)\s+procedente\s+(?:revocar|modificar)\b", re.I)
# Lo que dice que NO prospera.
_RX_CONFIRMA = re.compile(
    r"\bse\s+confirma\b|"
    r"\blo\s+procedente\s+es\s+confirmar\b|"
    r"\bprocede\s+confirmar\b|"
    r"\bdebe\s+confirmarse\b|"
    r"\b(?:es|resulta)\s+procedente\s+confirmar\b|"
    r"\bno\s+(?:procede|es\s+procedente|justifica|justifican|amerita|conduce\s+a)\s+"
    r"(?:revocar|revocarla|su\s+revocaci[óo]n|modificar|modificarla)\b", re.I)
# Cuando lo dice una PARTE, no el Tribunal: «la recurrente solicita que se
# revoque», «a juicio de la autoridad procede revocar».
#
# SUJETO Y VERBO, NO RAÍCES SUELTAS. La primera versión buscaba raíces como
# «plantea» o «argument» sin límite de palabra, y así tomaba por alegato de
# parte el cierre del propio Tribunal —«…fundados lo PLANTEADO, lo procedente
# es revocar»— y la síntesis —«sólo constituye un ARGUMENTO accesorio y no
# justifica revocarla»—. Exactamente las dos frases del 2/2026 que había que
# ver. Ahora hace falta que una parte procesal sea el sujeto y que el verbo sea
# de alegación, conjugado.
_RX_PARTE = r"(?:recurrente|autoridad\s+(?:recurrente|demandada)|quejos[oa]|actor[a]?\b|" \
            r"parte\s+(?:actora|demandada|recurrente|quejosa)|demandad[oa]|tercer[oa]\s+interesad[oa]|" \
            r"inconforme|promovente|revisionista)"
_RX_ALEGA = (r"\b(?:solicita|solicit[óo]|solicitan|pide|pidi[óo]|piden|pretende|pretendi[óo]|"
             r"sostiene|sostuvo|argumenta|argument[óo]|alega|aleg[óo]|afirma|afirm[óo]|"
             r"insiste|insisti[óo]|reclama|reclam[óo]|plantea|plante[óo]|aduce|adujo|"
             r"manifiesta|manifest[óo]|estima|estim[óo]|considera|consider[óo]|expresa|expres[óo])\b")
_RX_LO_DICE_UNA_PARTE = re.compile(
    r"(?:" + _RX_PARTE + r"[^.]{0,80}" + _RX_ALEGA + r"|" + _RX_ALEGA + r"[^.]{0,40}" + _RX_PARTE +
    r"|a\s+(?:su\s+)?juicio(?:\s+de\s+(?:la|el)\s+" + _RX_PARTE + r")?)[^.]{0,140}$", re.I)


def _zonas(texto: str) -> dict:
    """{estudio, resolutivos, sintesis} del texto plano del documento."""
    t = texto or ""
    i_res = t.find("R E S U E L V E")
    if i_res < 0:
        m = re.search(r"\n\s*RESUELVE\s*\n", t)
        i_res = m.start() if m else -1
    i_sin = t.find("SÍNTESIS", i_res if i_res >= 0 else 0)
    i_est = -1
    for rot in ("Estudio de los agravios", "Estudio de fondo", "Estudio de los conceptos",
                "Estudio del recurso", "Estudio."):
        i_est = t.find(rot)
        if i_est >= 0:
            break
    fin_est = i_res if i_res >= 0 else len(t)
    return {
        "estudio": t[max(i_est, 0):fin_est] if i_est >= 0 else t[:fin_est],
        "resolutivos": t[i_res:(i_sin if i_sin > i_res else len(t))] if i_res >= 0 else "",
        "sintesis": t[i_sin:] if i_sin >= 0 and i_sin > i_res else "",
    }


def _dichos(zona: str) -> list:
    """[(prospera: bool, frase)] de lo que el TRIBUNAL dice que se hace."""
    fuera = []
    for rx, prospera in ((_RX_REVOCA, True), (_RX_CONFIRMA, False)):
        for m in rx.finditer(zona):
            antes = zona[max(0, m.start() - 160):m.start()]
            if _RX_LO_DICE_UNA_PARTE.search(antes):
                continue
            ini = max(zona.rfind(".", 0, m.start()), zona.rfind("\n", 0, m.start())) + 1
            fin = zona.find(".", m.end())
            fuera.append((m.start(), prospera,
                          " ".join(zona[ini:(fin + 1 if fin > 0 else m.end() + 80)].split())))
    # EN EL ORDEN DEL TEXTO. La comprobación toma el primer dicho de los
    # resolutivos como el que manda, y agrupados por tipo el «primero» era
    # siempre el de revocar aunque el resolutivo empezara confirmando.
    return [(p_, f_) for _, p_, f_ in sorted(fuera)]


def contradicciones(texto: str, tipo_asunto: str) -> list:
    """Avisos de desenlace contradictorio en un recurso. Vacío si todo concuerda."""
    if _ta.normalizar(tipo_asunto) not in _TIPOS_CON_SENTENCIA_RECURRIDA:
        return []
    z = _zonas(texto)
    por_zona = {k: _dichos(v) for k, v in z.items() if v}
    # Lo que manda es lo que resuelve: los resolutivos. Si no hay resolutivos
    # legibles, no hay contra qué comparar.
    res = por_zona.get("resolutivos") or []
    if not res:
        return []
    va = res[0][0]
    malas = []
    for zona in ("estudio", "sintesis", "resolutivos"):
        for prospera, frase in por_zona.get(zona) or []:
            if prospera != va:
                malas.append((zona, frase))
    if not malas:
        return []
    que = "REVOCA" if va else "CONFIRMA"
    ejemplos = " · ".join(f"en {'la síntesis' if z_ == 'sintesis' else 'el ' + z_}: «{f[:170]}»"
                          for z_, f in malas[:3])
    return [
        f"NO FIRMABLE TAL COMO ESTÁ — EL PROYECTO SE CONTRADICE EN EL DESENLACE: "
        f"los resolutivos {que} y el texto dice lo contrario ({ejemplos}). Un "
        f"proyecto que confirma y revoca a la vez viola el principio de congruencia "
        f"y no se puede listar."]
