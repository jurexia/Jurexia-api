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

LA GUARDA PROCESAL (26-sep-2026, decisión 1 de David). En amparo directo una
VIOLACIÓN PROCESAL nunca queda sin materia, innecesaria ni caída con el
principal: los artículos 74, fracción V, y 174 de la Ley de Amparo mandan
decidirlas todas. La ÚNICA excepción es que un problema de FONDO prospere con
mayor beneficio que la reposición (artículo 189): entonces la procesal queda
innecesaria por mayor beneficio y su razón lo dice con el 189. Si lo que
prospera es una procesal, las demás procesales se deciden igual y el fondo
queda sin materia. Las piezas de la regla viven en `violacion_procesal`, que
también usa `modos_decision.repartir`.

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


def _vp():
    """El módulo de la violación procesal, o None si no carga. Sin él la
    guarda se apaga en vez de tumbar el reparto: es lo que pasaba ayer."""
    try:
        import violacion_procesal as _m
        return _m
    except Exception:                                   # pragma: no cover
        return None


def _decide(s: str) -> bool:
    """¿Esta calificación DECIDE el planteamiento? «innecesario» y «sin
    materia» no deciden: dicen por qué no se entra."""
    s = str(s or "").strip().lower().replace(" ", "_")
    return bool(s) and s not in (INNECESARIO, "sin_materia", "queda_sin_materia")


# El arranque con que este módulo escribe la caída con el principal. El
# estudio (`fase6_estudio`, al armar el criterio) lo reconoce por ESTE texto y
# lo redacta como «cae con el principal: no se estudia de fondo». Una procesal
# que vuelve de la pantalla con él —un reparto anterior al 26-sep— saldría sin
# decidir aunque su calificación diga «inoperante».
CAE_CON_PRINCIPAL = "Descansa en la premisa que se desestimó"


def _cae_con_principal(c) -> bool:
    """¿Viene calificado como caído con el principal? La fórmula Y una de las
    calificaciones con que el árbol la escribe; con otra calificación la
    fórmula es un resto, no una caída."""
    return (str(_get(c, "razonamiento", "") or "").startswith(CAE_CON_PRINCIPAL)
            and str(_get(c, "sentido", "") or "").strip().lower()
            in (INOPERANTE, "infundado", "ineficaz", "inatendible"))


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
            sentido_motor: str = "", tipo_asunto: str = "") -> tuple:
    """Ajusta EN SITIO el sentido de los accesorios que siguen al principal.

    `problemas`: los dicts de la fase 3, en su orden (pregunta, jerarquia,
    depende_de, clase). `criterios`: dicts o `Criterio` con problema/sentido/
    razonamiento/jerarquia; los que el secretario marcó llevan `tocado` True
    o están en `tocados`. `checklist`: la lista de la fase 5. `propuestas`:
    lo que propuso el motor (para `alcanza` y, en la guarda procesal, para
    devolverle a una procesal la calificación que el motor le dio).
    `tipo_asunto`: la guarda procesal es del amparo directo (sin tipo, rige).

    Devuelve (avisos, detalle): `detalle` es {problema: {"de": ..., "por_que":
    ...}} para que la pantalla diga de quién es cada calificación. Cuando
    actúa la guarda procesal, la entrada lleva además "guarda": "procesal"
    (se decide) o "mayor_beneficio_189" (la única excepción).
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

    # ═══ LA GUARDA PROCESAL (26-sep-2026) ══════════════════════════════════
    # David aprobó alinear el reparto con los artículos 74-V, 174 y 189: una
    # violación procesal se DECIDE siempre, y sólo deja de estudiarse cuando un
    # problema de FONDO prospera con mayor beneficio que la reposición. Antes
    # de hoy el árbol declaraba innecesaria la segunda procesal de un asunto
    # con el principal fundado, igual que cualquier accesorio: quedaba sin
    # decidir y nada lo decía (falla L2 del diagnóstico).
    #
    # La clase la da la fase 3 (`clase`); sin ella, el vocabulario de lo que se
    # combate, con la misma función que usa el resto del taller.
    _m = _vp()
    guarda = bool(_m and _m.guarda_aplica(tipo_asunto))
    _p_ref = por_texto.get(p_txt, (0, None))[1] or p_txt

    def _procesal(txt: str, pd) -> bool:
        return guarda and _m.clase_de(pd if pd is not None else txt) == "procesal"

    p_proc = _procesal(p_txt, por_texto.get(p_txt, (0, None))[1])
    # LA ÚNICA PUERTA PARA NO DECIDIR UNA PROCESAL: el principal es de fondo y
    # prospera entero. Si la concesión es para efectos que dan menos que
    # reponer, eso no lo sabe el código: lo sabe el secretario, y se le avisa.
    mb_fondo = bool(guarda and pros and alcanza
                    and _m.concesion_de_fondo_con_mayor_beneficio(_p_ref, p_sent, alcanza))
    # LO QUE EL MOTOR PROPUSO PARA ESTE PROBLEMA, POR SUS PROPIOS MÉRITOS.
    # Revisión del 26-sep-2026: la propuesta pasa por este mismo árbol y main
    # SOBRESCRIBE la propuesta guardada con lo que el árbol decide. Una
    # procesal que la excepción del 189 dejó «innecesaria» se guardaba así, y
    # si después el secretario cambiaba el principal, aquí ya no había qué
    # devolverle: quedaba sin calificar y el estudio la escribía «sin
    # materia» (se reprodujo con un fondo fundado que pasa a infundado). main
    # guarda ahora lo que el motor propuso ANTES del árbol en `sentido_propio`
    # y `razon_propia`; se prefiere eso, y sólo si no decide, lo guardado.
    def _propia(pr) -> tuple:
        for _ks, _kr in (("sentido_propio", "razon_propia"), ("sentido", "razon")):
            _s = str(_get(pr, _ks, "") or "").strip().lower().replace(" ", "_")
            _r = str(_get(pr, _kr, "") or "").strip()
            # Una propuesta guardada antes de hoy puede traer la caída que el
            # árbol le escribió («inoperante» + «Descansa en la premisa…»): eso
            # no es decidirla, es otra vez sacarla.
            if _decide(_s) and not _r.startswith(CAE_CON_PRINCIPAL):
                return _s, _r
        return "", ""

    _motor_de = {str(_get(pr, "problema", "")): _propia(pr) for pr in (propuestas or [])}
    por_189: list = []

    def _conservar(c, t: str) -> str:
        """Lo que se hace con una procesal que el árbol habría sacado: se
        queda con su calificación. Si la que trae no decide —llegó
        «innecesario» de otra pasada, o «inoperante» con la fórmula de caer
        con el principal—, se le devuelve la que el motor le propuso, con su
        razón; si tampoco hay, se queda como está y el aviso lo pide."""
        s_act = str(_get(c, "sentido", "")).strip().lower().replace(" ", "_")
        _cae = _cae_con_principal(c)
        if _decide(s_act) and not _cae:
            return s_act
        # La razón que traía sostenía NO decidirla —sin materia, innecesaria,
        # caída con el principal—: no se deja pegada a una calificación. El
        # estudio (fase6) reconoce la caída por su arranque y la escribiría
        # como tal.
        _set(c, "razonamiento", "")
        s_mot, r_mot = _motor_de.get(t, ("", ""))
        if s_mot:
            _set(c, "sentido", s_mot)
            if r_mot:
                _set(c, "razonamiento", r_mot)
            return s_mot
        return s_act

    def _aviso_procesal(t: str, s_kept: str, cae_con: bool, sugerida: str = "",
                        razon_sug: str = "") -> str:
        if not cae_con:
            # El mismo texto que el reparto global: en modo global corren
            # seguidos sobre el mismo problema y así no se lee dos veces.
            # «Principal procesal» quiere decir que PROSPERA: con uno
            # infundado el aviso decía «aunque el principal prospere».
            return _m.aviso_se_decide(t, s_kept, bool(p_proc and pros))
        _cal = (f"Se estudia con su calificación: {s_kept.replace('_', ' ')}."
                if _decide(s_kept) else
                "ESTÁ SIN CALIFICAR: califícala tú antes de generar.")
        _mot = (f" Para esta vía el motor le había escrito «{sugerida.replace('_', ' ')}»"
                + (f": {razon_sug[:160]}" if razon_sug else "")
                + ". Revisa que su calificación siga siendo la tuya."
                if sugerida else "")
        return (f"«{t[:90]}» es una VIOLACIÓN PROCESAL y NO se declaró caída con "
                f"el principal: los artículos 74, fracción V, y 174 de la Ley de "
                f"Amparo mandan decidirla por lo que ella misma plantea. {_cal}{_mot}")

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
        proc = _procesal(t, pdict)

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
            # SU MARCA MANDA, PERO UNA PROCESAL SIN DECIDIR SE LE DICE. Si él
            # mismo la puso innecesaria sin que el fondo prospere con mayor
            # beneficio, el proyecto dejaría una violación procesal sin
            # decidir: no se pisa, se avisa.
            if proc and not _decide(str(_get(c, "sentido", ""))) \
                    and str(_get(c, "sentido", "")).strip() and not mb_fondo:
                avisos.append(
                    f"«{t[:90]}» la marcaste {str(_get(c, 'sentido', '')).replace('_', ' ')} "
                    f"y es una VIOLACIÓN PROCESAL: los artículos 74, fracción V, y 174 de "
                    f"la Ley de Amparo mandan decidirla, y sólo puede quedar innecesaria "
                    f"si un problema de fondo prospera con mayor beneficio que la "
                    f"reposición (artículo 189). Se respeta tu marca; revisa si es lo "
                    f"que quieres.")
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
            if s == INNECESARIO and proc:
                # ── LA GUARDA, CUANDO EL PRINCIPAL PROSPERA ──
                if mb_fondo:
                    # La única excepción, y dicha con su artículo.
                    _set(c, "sentido", INNECESARIO)
                    _set(c, "razonamiento", _m.RAZON_MAYOR_BENEFICIO)
                    detalle[t] = {"de": "principal", "guarda": "mayor_beneficio_189",
                                  "por_que": "violación procesal innecesaria por mayor beneficio "
                                             "(art. 189): el fondo que prospera da más que reponer"}
                    por_189.append(t)
                else:
                    _k = _conservar(c, t)
                    detalle[t] = {"de": "propio", "guarda": "procesal",
                                  "por_que": "violación procesal: los arts. 74-V y 174 mandan "
                                             "decidirla; no queda sin materia"}
                    avisos.append(_aviso_procesal(t, _k, cae_con=False))
                    siguen += 1
                continue
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
            if s in (INOPERANTE, "infundado", "ineficaz", "inatendible", INNECESARIO) and proc:
                # ── LA GUARDA, CUANDO EL PRINCIPAL NO PROSPERA ──
                # «Descansa en la premisa desestimada, de modo que su estudio no
                # produciría ningún fin práctico» es no decidirla. Una procesal
                # se decide por lo que ella plantea; lo que el motor escribió
                # para esta vía se le enseña al secretario, no se impone.
                _k = _conservar(c, t)
                detalle[t] = {"de": "propio", "guarda": "procesal",
                              "por_que": "violación procesal: se decide por sí misma "
                                         "(arts. 74-V y 174); no cae con el principal"}
                avisos.append(_aviso_procesal(
                    t, _k, cae_con=True,
                    sugerida=(s if _entrada_tiene_suerte(entrada, sentido_motor) else ""),
                    razon_sug=razon))
                siguen += 1
                continue
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

    # ── NINGUNA PROCESAL SALE SIN DECIDIR, VENGA DE DONDE VENGA ──────────
    # El árbol ya no las saca; pero pueden LLEGAR sacadas: de un reparto
    # anterior, de la propuesta, de una sesión de antes del 26-sep. Las que el
    # árbol dejó en paz (tema distinto, sin dependencia, lo fundado no alcanza)
    # conservaban ese «innecesario» sin que nada lo mirara.
    if guarda:
        for c in criterios:
            if c is principal or str(_get(c, "jerarquia", "")).lower() == "principal":
                continue
            t = str(_get(c, "problema", ""))
            if t in _tocados or not _procesal(t, por_texto.get(t, (0, None))[1]):
                continue
            s_act = str(_get(c, "sentido", "")).strip().lower()
            _cae = _cae_con_principal(c)
            if not s_act or (_decide(s_act) and not _cae):
                continue
            if (detalle.get(t) or {}).get("guarda") == "procesal":
                continue                   # ya se avisó dentro del recorrido
            if _cae:
                # Llegó «caída con el principal»: se decide por sí misma.
                _k = _conservar(c, t)
                detalle[t] = {"de": "propio", "guarda": "procesal",
                              "por_que": "violación procesal: se decide por sí misma "
                                         "(arts. 74-V y 174); no cae con el principal"}
                avisos.append(_aviso_procesal(t, _k, cae_con=True))
                continue
            if mb_fondo:
                # «sin materia» se escribe como tal en el estudio, que sólo
                # reconoce «innecesario»; la razón es la del 189.
                _set(c, "sentido", INNECESARIO)
                if "189" not in str(_get(c, "razonamiento", "") or ""):
                    _set(c, "razonamiento", _m.RAZON_MAYOR_BENEFICIO)
                if (detalle.get(t) or {}).get("guarda") != "mayor_beneficio_189":
                    detalle[t] = {"de": "principal", "guarda": "mayor_beneficio_189",
                                  "por_que": "violación procesal innecesaria por mayor beneficio "
                                             "(art. 189): el fondo que prospera da más que reponer"}
                    por_189.append(t)
                continue
            _k = _conservar(c, t)
            detalle[t] = {"de": "propio", "guarda": "procesal",
                          "por_que": "violación procesal: los arts. 74-V y 174 mandan "
                                     "decidirla; no queda sin materia"}
            avisos.append(_aviso_procesal(t, _k, cae_con=False))

    if por_189:
        avisos.append(
            f"VIOLACIÓN(ES) PROCESAL(ES) INNECESARIA(S) POR MAYOR BENEFICIO "
            f"(artículo 189 de la Ley de Amparo): "
            + " · ".join(f"«{x[:80]}»" for x in por_189[:4])
            + f". El principal es de fondo y resulta {p_sent.replace('_', ' ')}, y se "
              f"entiende que esa concesión da a la parte quejosa más que reponer el "
              f"procedimiento. Los artículos 74, fracción V, y 174 mandan decidir "
              f"todas las procesales y ésta es la única excepción: si la concesión "
              f"de fondo no da más que la reposición —por ejemplo, porque es para "
              f"efectos—, márcala tú y se estudia.")
    # EL FONDO, CON LA REPOSICIÓN. Si lo que prospera es una procesal, la
    # sentencia reclamada se deja insubsistente y el fondo queda sin materia;
    # el árbol lo aplica a lo que depende del principal, pero no inventa
    # dependencias donde la fase 5 no las escribió. Lo que quedó «por su
    # cuenta» se le señala al secretario, que es quien sabe si da más.
    if guarda and pros and alcanza and p_proc:
        _fondo_vivo = [
            str(_get(c, "problema", "")) for c in criterios
            if c is not principal
            and str(_get(c, "jerarquia", "")).lower() != "principal"
            and str(_get(c, "problema", "")) not in _tocados
            and not _procesal(str(_get(c, "problema", "")),
                              por_texto.get(str(_get(c, "problema", "")), (0, None))[1])
            and (detalle.get(str(_get(c, "problema", ""))) or {}).get("de") in ("propio", "distinto")
            and _decide(str(_get(c, "sentido", "")))]
        if _fondo_vivo:
            avisos.append(
                "EL PRINCIPAL ES UNA VIOLACIÓN PROCESAL QUE PROSPERA: la reposición "
                "deja insubsistente la sentencia reclamada y el fondo queda sin "
                "materia, salvo lo que dé más que reponer. Siguen calificados por su "
                "cuenta: " + " · ".join(f"«{x[:80]}»" for x in _fondo_vivo[:4])
                + ". Si no piden más que la reposición, márcalos innecesarios.")

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


def _entrada_tiene_suerte(entrada: dict, sentido_motor: str = "") -> bool:
    """¿El motor escribió una suerte para la vía «no prospera»? (para no
    presentarle al secretario como «lo que escribió el motor» lo que en
    realidad dedujo el árbol de un `depende_de`)."""
    return bool(_suerte(entrada or {}, "si_no_prospera", sentido_motor)[0])


def reparto_para_pantalla(problemas: list, criterios: list, checklist: list = None,
                          propuestas: list = None, sentido_motor: str = "",
                          tipo_asunto: str = "") -> dict:
    """Lo que devuelve /taller/reparto: los criterios ya ajustados y de quién
    es cada uno. Trabaja sobre COPIAS: la pantalla decide qué hace con ello."""
    copia = [dict(c) if isinstance(c, dict) else {
        "problema": _get(c, "problema", ""), "sentido": _get(c, "sentido", ""),
        "razonamiento": _get(c, "razonamiento", ""),
        "jerarquia": _get(c, "jerarquia", "accesorio"),
        "tocado": bool(_get(c, "tocado", False))} for c in (criterios or [])]
    avisos, detalle = aplicar(problemas, copia, checklist, propuestas,
                              sentido_motor=sentido_motor, tipo_asunto=tipo_asunto)
    for c in copia:
        d = detalle.get(str(c.get("problema", "")), {})
        c["de"] = d.get("de", "")
        c["por_que"] = d.get("por_que", "")
        # «procesal» (se decide) o «mayor_beneficio_189»: la pantalla de hoy
        # pinta `de` y `por_que`; esto viaja para la que lo distinga.
        c["guarda"] = d.get("guarda", "")
    return {"criterios": copia, "avisos": avisos}
