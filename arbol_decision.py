"""LA SUERTE DE LOS ACCESORIOS LA DICTA EL PRINCIPAL — en los tres modos.

POR QUÉ EXISTE. Amparo directo 93/2026, 22-sep-2026. David: «la lógica del
planteamiento principal no domina la suerte de los planteamientos secundarios
en automático (…) si cambio de sentido o el sentido de la resolución principal
es uno, los accesorios caen por su propio peso cuando tienen estrecha relación
o cuando a ningún fin práctico produce su análisis».

LO QUE HABÍA. La sustracción de materia vivía en `modos_decision.repartir` y
sólo en el modo GLOBAL, sólo cuando el principal es fundado. En el modo por
problema —el que se usa al corregir un tema— cada planteamiento era una isla.
Y `depende_de`, que la fase 3 calcula para cada problema, no lo leía nadie.

LA REGLA, en las dos direcciones:

  · EL PRINCIPAL PROSPERA y alcanza → los accesorios que dependen de él quedan
    SIN MATERIA («innecesario»): se dice y se dice por qué.
  · EL PRINCIPAL NO PROSPERA → los accesorios que descansaban en su premisa
    caen con él: son INOPERANTES, porque parten de lo que ya se desestimó
    («la Sala debía estudiar los alegatos contra el crédito» presupone que el
    crédito estaba en la litis, y eso acaba de negarse).

Y LOS ESCAPES, que son los de siempre y uno nuevo:
  · lo que el secretario marcó a mano (`tocado`) no se toca;
  · el tema DISTINTO —el motor lo declaró ajeno a la suerte del principal—;
  · el que pide MAYOR BENEFICIO que lo concedido en el principal;
  · el que no depende de nada: se estudia por su cuenta.

DE DÓNDE SALE LA DEPENDENCIA. De tres fuentes, en este orden: la suerte
condicional que la fase 5 escribe por tema (`si_prospera` / `si_no_prospera`,
con su razón), la `relacion` que declara, y el `depende_de` de la fase 3. No
hay heurística de texto: una palabra dentro de una pregunta no prueba
dependencia, y este proyecto ya sabe qué pasa con las heurísticas de una
palabra.

Un solo módulo, una sola regla, aplicada en tres sitios: al proponer (para que
la pantalla ya enseñe la suerte), en /taller/reparto (cuando el secretario
cambia el principal) y al resolver (para que el documento la lleve).
"""
from __future__ import annotations

INNECESARIO = "innecesario"
INOPERANTE = "inoperante"

# Lo que un accesorio pide y que NO se declara innecesario aunque el principal
# prospere: da más de lo concedido. Es la misma lista de modos_decision, que
# sigue siendo su dueño; se lee de ahí para que no haya dos.
try:
    from modos_decision import _pide_mas as pide_mas   # noqa: F401
except Exception:                                      # pragma: no cover
    def pide_mas(problema: str) -> bool:
        return False

_SENTIDOS = ("fundado", "esencialmente_fundado", "sustancialmente_fundado",
             "parcialmente_fundado", "fundado_insuficiente", "infundado",
             "inoperante", "inatendible", "ineficaz", "sin_materia", INNECESARIO)


def _get(x, k, d=""):
    if isinstance(x, dict):
        return x.get(k, d)
    return getattr(x, k, d)


def _set(x, k, v):
    if isinstance(x, dict):
        x[k] = v
    else:
        setattr(x, k, v)


def _texto(p) -> str:
    return p if isinstance(p, str) else str((p or {}).get("pregunta") or p)


def _prospera(s: str) -> bool:
    try:
        import tipos_asunto as _ta
        return _ta.prospera(s)
    except Exception:
        s = (s or "").lower()
        return "fundad" in s and "insuficien" not in s


def _sentido_valido(s: str) -> str:
    s = str(s or "").strip().lower().replace(" ", "_")
    if s in ("sin_materia", "queda_sin_materia"):
        return INNECESARIO
    return s if s in _SENTIDOS else ""


# ═══ LA SUERTE CONDICIONAL, LEÍDA DE LA LISTA DE COMPROBACIÓN ══════════════

def entrada_de(checklist: list, numero: int, problema: str = "") -> dict:
    """La entrada de la lista de comprobación del problema `numero` (1..n)."""
    for c in (checklist or []):
        if not isinstance(c, dict):
            continue
        try:
            if int(c.get("numero")) == numero:
                return c
        except (TypeError, ValueError):
            pass
    if problema:
        _p = problema.strip().lower()[:60]
        for c in (checklist or []):
            if isinstance(c, dict) and _p and _p in str(c.get("tema") or "").lower():
                return c
    return {}


import re as _re

# «Fundado en lo conducente: …», «Infundado: al no formar parte…»,
# «Inoperante, porque…», «Queda sin materia»: el sentido va al principio de la
# suerte escrita en prosa, y el resto es la razón.
_RX_SUERTE = _re.compile(
    r"^\W*(?:(?:es|resulta|ser[íi]a|quedar[íi]a|queda)\s+)?"
    r"((?:esencialmente|sustancialmente|parcialmente)\s+fundad[oa]s?|"
    r"fundad[oa]s?\s+(?:pero|aunque)\s+insuficiente|"
    r"fundad[oa]s?|infundad[oa]s?|inoperantes?|ineficaz|ineficaces|inatendibles?|"
    r"innecesari[oa]s?|sin\s+materia)\b\s*(?:en\s+lo\s+conducente)?[\s:,.;—–-]*(.*)$",
    _re.I | _re.S)


def _leer_suerte(texto: str) -> tuple:
    """(sentido, razón) leídos de una suerte escrita en prosa; ('', '') si no
    empieza por una calificación."""
    m = _RX_SUERTE.match(str(texto or "").strip())
    if not m:
        return "", ""
    cab = m.group(1).lower()
    cab = _re.sub(r"\s+", " ", cab)
    if "insuficiente" in cab:
        s = "fundado_insuficiente"
    elif cab.startswith(("esencialmente", "sustancialmente", "parcialmente")):
        s = cab.split()[0] + "_fundado"
    elif cab.startswith("fundad"):
        s = "fundado"
    elif cab.startswith("infundad"):
        s = "infundado"
    elif cab.startswith("inoperante"):
        s = INOPERANTE
    elif cab.startswith("ineficac"):
        s = "ineficaz"
    elif cab.startswith("inatendible"):
        s = "inatendible"
    else:
        s = INNECESARIO
    return s, m.group(2).strip()


def _suerte(entrada: dict, clave: str, sentido_motor: str = "") -> tuple:
    """(sentido, razón) de `si_prospera` / `si_no_prospera`.

    Primero lo estructurado que la fase 5 escribe desde hoy. Para las listas
    de antes —`con_propuesta` / `con_alternativa`, en prosa y referidas a la
    vía del motor— se traduce: si la propuesta del motor prosperaba,
    `con_propuesta` es la suerte «si prospera» y `con_alternativa` la otra;
    si no, al revés. Medido en el 93/2026: el motor marcó el tema 2 como
    DISTINTO y a la vez escribió que, en la vía contraria, «al no formar parte
    de la litis el crédito fiscal, los alegatos no podían ampliarla»: su
    suerte depende del principal aunque la etiqueta diga lo contrario. La
    suerte escrita es más concreta que la etiqueta, y manda.
    """
    s = entrada.get(clave)
    if isinstance(s, dict):
        return _sentido_valido(s.get("sentido")), str(s.get("razon") or "").strip()
    if isinstance(s, str) and s.strip():
        return _leer_suerte(s)
    if not sentido_motor:
        return "", ""
    motor_prospera = _prospera(sentido_motor)
    quiero_prospera = (clave == "si_prospera")
    fuente = "con_propuesta" if (motor_prospera == quiero_prospera) else "con_alternativa"
    txt = str(entrada.get(fuente) or "").strip()
    if not txt or txt.upper().startswith("SIN DETERMINAR"):
        return "", ""
    return _leer_suerte(txt)


def relacion_de(problema, principal_num: int, entrada: dict) -> str:
    """'distinto' | 'depende' | '' (no declarada)."""
    if entrada.get("tema_distinto"):
        return "distinto"
    rel = str(entrada.get("relacion") or "").strip().lower()
    if rel in ("distinto", "independiente"):
        return "distinto"
    if rel in ("depende", "dependiente", "accesorio"):
        return "depende"
    if isinstance(problema, dict):
        dep = problema.get("depende_de")
        try:
            if dep is not None and int(dep) == int(principal_num):
                return "depende"
        except (TypeError, ValueError):
            pass
    return ""


# ═══ LA REGLA ══════════════════════════════════════════════════════════════

def aplicar(problemas: list, criterios: list, checklist: list = None,
            propuestas: list = None, tocados: set = None,
            sentido_motor: str = "") -> tuple:
    """Ajusta EN SITIO el sentido de los accesorios que siguen al principal.

    `problemas`: los dicts de la fase 3, en su orden (pregunta, jerarquia,
    depende_de). `criterios`: dicts o `Criterio` con problema/sentido/
    razonamiento/jerarquia; los que el secretario marcó llevan `tocado` True
    o están en `tocados`. `checklist`: la lista de la fase 5. `propuestas`:
    lo que propuso el motor (para `alcanza`).

    Devuelve (avisos, detalle): `detalle` es {problema: {"de": ..., "por_que":
    ...}} para que la pantalla diga de quién es cada calificación.
    """
    avisos: list = []
    detalle: dict = {}
    if not criterios:
        return avisos, detalle
    _tocados = set(tocados or [])
    for c in criterios:
        if _get(c, "tocado", False):
            _tocados.add(str(_get(c, "problema", "")))

    textos = [_texto(p) for p in (problemas or [])]
    por_texto = {t: (i + 1, p) for i, (t, p) in enumerate(zip(textos, problemas or []))}

    principal = next((c for c in criterios
                      if str(_get(c, "jerarquia", "")).lower() == "principal"), None)
    if principal is None:
        # Sin marca de jerarquía, el primero de la fase 3 es el principal.
        principal = next((c for c in criterios
                          if textos and str(_get(c, "problema", "")) == textos[0]),
                         criterios[0])
    p_txt = str(_get(principal, "problema", ""))
    p_num = por_texto.get(p_txt, (1, None))[0]
    p_sent = str(_get(principal, "sentido", "")).strip().lower()
    if not p_sent:
        return avisos, detalle
    pros = _prospera(p_sent)
    detalle[p_txt] = {"de": "principal", "por_que": ""}

    # ¿ALCANZA? Un principal fundado que sólo repone el procedimiento no deja
    # sin materia lo que pide el fondo. La propuesta lo dice cuando lo sabe.
    alcanza = True
    for pr in (propuestas or []):
        if str(_get(pr, "problema", "")) == p_txt and _get(pr, "alcanza", True) is False:
            alcanza = False
    if pros and not alcanza:
        avisos.append(
            "NO SE APLICÓ LA SUSTRACCIÓN DE MATERIA: el motor no pudo afirmar "
            "que lo fundado del principal alcance para resolver el asunto. Los "
            "accesorios se estudian.")

    caen, siguen = 0, 0
    for c in criterios:
        if c is principal:
            continue
        t = str(_get(c, "problema", ""))
        if str(_get(c, "jerarquia", "")).lower() == "principal":
            continue
        num, pdict = por_texto.get(t, (0, None))
        entrada = entrada_de(checklist, num, t)
        rel = relacion_de(pdict, p_num, entrada)
        suyo = t in _tocados

        if suyo:
            detalle[t] = {"de": "tuya", "por_que": "lo calificaste tú; manda sobre la suerte del principal"}
            # Si su marca ya no casa con lo que el principal implica, se avisa,
            # no se pisa.
            _esp, _ = _suerte(entrada, "si_prospera" if pros else "si_no_prospera", sentido_motor)
            if _esp and _esp != str(_get(c, "sentido", "")).lower():
                avisos.append(
                    f"«{t[:70]}» lo calificaste {str(_get(c, 'sentido', '')).replace('_', ' ')}; "
                    f"con el principal {p_sent.replace('_', ' ')} la suerte que el motor "
                    f"le había escrito era {_esp.replace('_', ' ')}. Se respeta tu marca.")
            continue
        # LA SUERTE ESCRITA MANDA SOBRE LA ETIQUETA «DISTINTO». Si el motor dijo
        # que el tema es distinto pero escribió que en esta vía «al no formar
        # parte de la litis… no podían», lo concreto es la suerte.
        _s_via, _ = _suerte(entrada, "si_prospera" if pros else "si_no_prospera", sentido_motor)
        if rel == "distinto" and not _s_via:
            detalle[t] = {"de": "distinto", "por_que": "tema distinto: no cuelga del principal y se estudia aparte"}
            siguen += 1
            continue
        if pros and pide_mas(t):
            detalle[t] = {"de": "mayor_beneficio", "por_que": "pide más que lo concedido en el principal: se estudia"}
            avisos.append(
                f"NO SE DECLARÓ INNECESARIO «{t[:90]}»: pide algo que da MÁS que lo "
                f"concedido en el principal. Declararlo innecesario sería negarlo "
                f"sin decirlo. Se estudia.")
            continue

        if pros:
            if not alcanza:
                continue
            s, razon = _suerte(entrada, "si_prospera", sentido_motor)
            if not s and rel == "depende":
                s = INNECESARIO
            if s == INNECESARIO:
                _set(c, "sentido", INNECESARIO)
                _set(c, "razonamiento",
                     "Dado el sentido del estudio del problema principal, queda sin "
                     "materia el análisis de este planteamiento"
                     + (f": {razon}" if razon else "."))
                detalle[t] = {"de": "principal", "por_que": razon or "queda sin materia al prosperar el principal"}
                caen += 1
            elif s:
                # El motor escribió otra suerte para esta vía (p. ej. «fundado en
                # lo conducente»): se respeta, con su razón.
                _set(c, "sentido", s)
                if razon:
                    _set(c, "razonamiento", razon)
                detalle[t] = {"de": "principal", "por_que": razon}
                siguen += 1
            else:
                detalle[t] = {"de": "propio", "por_que": "no consta que dependa del principal: se estudia por su cuenta"}
                siguen += 1
        else:
            s, razon = _suerte(entrada, "si_no_prospera", sentido_motor)
            if not s and rel == "depende":
                s = INOPERANTE
            if s in (INOPERANTE, "infundado", "ineficaz", "inatendible", INNECESARIO):
                if s == INNECESARIO:
                    s = INOPERANTE
                _set(c, "sentido", s)
                _set(c, "razonamiento",
                     "Descansa en la premisa que se desestimó al resolver el "
                     "problema principal"
                     + (f": {razon}" if razon else
                        ", de modo que su estudio no produciría ningún fin práctico."))
                detalle[t] = {"de": "principal", "por_que": razon or "cae con el principal: parte de una premisa desestimada"}
                caen += 1
            elif s:
                _set(c, "sentido", s)
                if razon:
                    _set(c, "razonamiento", razon)
                detalle[t] = {"de": "principal", "por_que": razon}
                siguen += 1
            else:
                detalle[t] = {"de": "propio", "por_que": "no consta que dependa del principal: se estudia por su cuenta"}
                siguen += 1

    if caen:
        if pros:
            avisos.append(
                f"SUSTRACCIÓN DE MATERIA aplicada a {caen} planteamiento(s): al "
                f"resultar {p_sent.replace('_', ' ')} el principal, su estudio "
                f"queda sin materia. El proyecto lo DICE, no lo calla.")
        else:
            avisos.append(
                f"{caen} planteamiento(s) accesorio(s) CAEN CON EL PRINCIPAL: al "
                f"resultar {p_sent.replace('_', ' ')}, descansan en una premisa "
                f"desestimada y se declaran inoperantes con esa razón. Si alguno "
                f"merece estudio propio, márcalo tú.")
    return avisos, detalle


def reparto_para_pantalla(problemas: list, criterios: list, checklist: list = None,
                          propuestas: list = None, sentido_motor: str = "") -> dict:
    """Lo que devuelve /taller/reparto: los criterios ya ajustados y de quién
    es cada uno. Trabaja sobre COPIAS: la pantalla decide qué hace con ello."""
    copia = [dict(c) if isinstance(c, dict) else {
        "problema": _get(c, "problema", ""), "sentido": _get(c, "sentido", ""),
        "razonamiento": _get(c, "razonamiento", ""),
        "jerarquia": _get(c, "jerarquia", "accesorio"),
        "tocado": bool(_get(c, "tocado", False))} for c in (criterios or [])]
    avisos, detalle = aplicar(problemas, copia, checklist, propuestas,
                              sentido_motor=sentido_motor)
    for c in copia:
        d = detalle.get(str(c.get("problema", "")), {})
        c["de"] = d.get("de", "")
        c["por_que"] = d.get("por_que", "")
    return {"criterios": copia, "avisos": avisos}
