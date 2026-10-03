# -*- coding: utf-8 -*-
"""LA PROPUESTA SIEMPRE ELIGE UN LADO: EL QUE PASA DEL 50% (David, 2-oct-2026).

«Si hay un 50.01% de probabilidad hacia un lado sea esa la propuesta de
resolución, para que el secretario pueda generar el proyecto en automático.»

DE DÓNDE SALE EL NÚMERO (David, 2-oct-2026, corrigiendo la primera versión):
«Tú partes de que es con la tasa del tribunal y no es así; quizá el tribunal se
equivoca. El tribunal sólo sirve como guía en casos de precedentes claros, pero
el motor puede aprovechar su capacidad de razonar una respuesta en aplicación
de una jurisprudencia aplicable al caso, una aplicable temáticamente o, incluso,
una por analogía (…) lo que resuelve es la inteligencia en el razonamiento
jurídico». Así que:

 · LA PROBABILIDAD ES RAZONADA, no estadística: la da el EXAMINADOR
   (`examinador.py`), un segundo razonador que examina las dos vías que el
   motor ya escribió y dice con qué probabilidad prosperan los conceptos o
   agravios. Si no llega, manda el lado que razonó el motor.
 · NI LA TASA DEL TRIBUNAL NI SUS PRECEDENTES DECIDEN. Los precedentes del
   propio tribunal sobre el mismo problema sólo AVISAN al secretario
   (`aviso_precedentes`): un caso muy parecido no prueba que se resolvió bien.
   La primera versión de este módulo mezclaba la tasa (71% de negativas en
   amparo directo) y proponía negar siempre; se retiró el mismo día.

LO QUE HACE AQUÍ: la mecánica pura —qué lado, si se voltea, cómo se
intercambian las vías— para probarla sin red; main pone el examen alrededor.
"""
from __future__ import annotations

import re

# Las calificaciones que hacen PROSPERAR un planteamiento; las demás no. «Fundado
# pero insuficiente / pero inoperante» no prospera: tiene razón y no alcanza.
_RX_NO = re.compile(r"infundad|insuficient|inoperant|ineficaz|inatendibl|improceden", re.I)
_RX_SI = re.compile(r"fundad", re.I)
_RX_SIN = re.compile(r"no se estudi|sin materia|innecesari|desech|sobres", re.I)


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


def voto_motor(sentido) -> int | None:
    """El lado que razonó el motor (su sentido global): 1 prospera, 0 no."""
    return prospera(sentido)


def aviso_precedentes(filas: list, sentido: str) -> str:
    """El aviso de los precedentes del PROPIO tribunal sobre el mismo problema
    (filas de `fase_oaj`, nivel «mismo_problema»): guía, nunca decide. «» si no
    hay ninguno de ese nivel con calificación."""
    quiere = prospera(sentido)
    vistos, a_favor, en_contra = set(), [], []
    for f in filas or []:
        if not isinstance(f, dict) or str(f.get("nivel") or "") != "mismo_problema":
            continue
        y = prospera(f.get("calificacion"))
        clave = (f.get("neun"), f.get("expediente"))
        if y is None or clave in vistos:
            continue
        vistos.add(clave)
        etq = f"{f.get('tipo_asunto') or ''} {f.get('expediente') or ''}".strip() or "un asunto"
        etq += f" ({f.get('similitud')}% mismo problema, «{f.get('calificacion')}»)"
        (a_favor if (quiere is not None and y == quiere) else en_contra).append(etq)
    if not (a_favor or en_contra):
        return ""
    partes = []
    if en_contra:
        partes.append("lo resolvió al revés en " + "; ".join(en_contra[:3]))
    if a_favor:
        partes.append("lo resolvió en el mismo sentido en " + "; ".join(a_favor[:3]))
    return ("PRECEDENTES DEL PROPIO TRIBUNAL (guía, no deciden): sobre este mismo problema, el tribunal "
            + " y ".join(partes) + ". Un caso muy parecido no prueba que se haya resuelto bien: revísalos "
            "antes de firmar.")


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


def aplicar(glob, propuestas: list, problemas: list, prob: dict | None,
            jerarquias: dict | None = None) -> dict:
    """Fija EN SITIO el lado de la propuesta con la probabilidad RAZONADA.

    `glob`: la `Global` de la fase 5 (o un dict con sus campos); `propuestas`:
    las `Propuesta` ya emparejadas con `problemas`. `prob`: {p_prospera, lado
    («prospera» | «no_prospera»), fuente, explicacion} — la del examinador; si
    es None o no trae lado, manda el lado que razonó el motor.

    Devuelve {probabilidad, principal, volteada, necesita_razon, sentido,
    avisos, fija_ejecutoria}. `necesita_razon`: el lado ganador no tiene razón
    escrita y main la pide. `fija_ejecutoria`: el principal lo fijó la regla de
    la ejecutoria (`origen` «ejecutoria») y el examen no lo voltea. Nunca lanza
    por datos raros: lo que no entiende, lo deja."""
    avisos: list = []
    i = indice_principal(problemas, propuestas, jerarquias)
    pral = propuestas[i] if i >= 0 else None
    s_glob = str(_g(glob, "sentido", "") or "").strip().lower()
    s_pral = str(_g(pral, "sentido", "") or "").strip().lower() if pral is not None else ""
    s_motor = s_glob or s_pral
    prob = dict(prob or {})
    if prob.get("lado") not in ("prospera", "no_prospera"):
        # SIN EXAMEN, manda el motor: su lado, sin número que inventar.
        _vm = voto_motor(s_motor) if s_motor else None
        prob = {"p_prospera": None, "fuente": "motor",
                "lado": None if _vm is None else ("prospera" if _vm else "no_prospera"),
                "explicacion": "Se propone el lado que razonó el motor; el examen de las dos vías no llegó."}
    prob["sentido_motor"] = s_motor
    prob["volteada"] = False
    info = {"probabilidad": prob, "principal": i, "volteada": False,
            "necesita_razon": False, "sentido": s_glob, "avisos": avisos,
            "fija_ejecutoria": False}
    lado = prob.get("lado")
    if lado not in ("prospera", "no_prospera"):
        return info
    quiere = lado == "prospera"
    v = voto_motor(s_motor) if s_motor else None

    # LO QUE FIJÓ LA EJECUTORIA NO SE VOLTEA (revisión adversarial, 3-oct-2026).
    # En amparo directo contra una sentencia dictada en cumplimiento, la regla
    # de la ejecutoria (`cumplimiento_ejecutoria.aplicar`, en main ANTES de
    # esto) deja el principal «inoperante» por vinculado y main lo marca con
    # `origen` «ejecutoria». El volteo lo sobrescribía a «fundado» con la razón
    # de la vía contraria; en el resolver la misma regla lo volvía a
    # «inoperante», `desenlace.reconciliar` a «fundado», y el proyecto salía
    # concediendo sobre lo que la ejecutoria dejó firme con la razón de la
    # inoperancia: resolutivo y estudio incongruentes. Si para voltear hay que
    # tocar ese principal, no se voltea: se queda el lado del motor y se avisa.
    # (Si el examen va al lado del principal fijado, el volteo sí corre: no lo
    # toca.)
    _vp = voto_motor(s_pral) if s_pral else None
    if (pral is not None and str(_g(pral, "origen", "") or "") == "ejecutoria"
            and not (v is not None and bool(v) == quiere)
            and (_vp is None or bool(_vp) != quiere)):
        _lado_m = v if v is not None else _vp
        _pe = prob.get("p_prospera")
        prob.update({
            # «motor» porque es su lado (la pantalla sólo conoce examinador,
            # jurimetria y motor); `fijada_por` dice por qué no mandó el examen.
            "examen_lado": lado, "examen_p": _pe, "p_prospera": None, "fuente": "motor",
            "fijada_por": "ejecutoria",
            "lado": None if _lado_m is None else ("prospera" if _lado_m else "no_prospera"),
            "explicacion": ("Se propone el lado que razonó el motor: el problema principal lo fijó la "
                            "ejecutoria que se cumplimenta y el examen de las dos vías no lo voltea.")})
        info["fija_ejecutoria"] = True
        _pct = ""
        if isinstance(_pe, (int, float)):
            _pct = f" ({int(round((_pe if quiere else 1 - _pe) * 100))}%)"
        avisos.append(
            "EL PRINCIPAL LO FIJA LA EJECUTORIA: el examen de las dos vías se inclinaba por que los "
            f"conceptos o agravios {'prosperen' if quiere else 'no prosperen'}{_pct}, pero no voltea lo que "
            f"la ejecutoria dejó vinculado: se queda el lado del motor («{s_motor.replace('_', ' ')}»). "
            "Si la ejecutoria no vinculaba ese punto, corrígelo en «De dónde viene lo reclamado».")
        if not s_glob and s_pral:
            # La global cuelga del principal, como cuando el motor ya estaba
            # del lado que gana.
            _s(glob, "sentido", s_pral)
            _s(glob, "razon", str(_g(pral, "razon", "") or ""))
            _s(glob, "apoyos", list(_g(pral, "apoyos", None) or []))
            if not str(_g(glob, "problema_que_decide", "") or "").strip():
                _s(glob, "problema_que_decide", str(_g(pral, "problema", "") or ""))
        info["sentido"] = str(_g(glob, "sentido", "") or "")
        _s(glob, "alcanza", bool(info["sentido"]))
        return info

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
    # TODO CAMBIO DE SENTIDO SOBRE UNO QUE EL MOTOR SÍ ESCRIBIÓ ES UN VOLTEO
    # (revisión adversarial, 3-oct-2026): desde un global «sin materia»
    # (`prospera` None) se pasaba a «fundado» sin aviso ni la línea «el motor
    # leía…». Sólo sin ningún sentido del motor es «el motor no propuso».
    volteada = bool(s_motor)
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
        avisos.append("EL EXAMEN DE LAS DOS VÍAS VOLTEÓ LA PROPUESTA: " + str(prob.get("explicacion", "")))
    elif not s_motor:
        avisos.append("El motor no propuso ningún sentido; se propone el más probable según el examen: "
                      + str(prob.get("explicacion", "")))
    return info


# ═══════════════════════════════════════════════════════════════════════════
# LOS AVISOS DE REGISTROS, REHECHOS TRAS EL VOLTEO (revisión adversarial,
# 3-oct-2026). `fase5_propuesta.proponer` los escribe ANTES del volteo:
# «La propuesta del asunto se apoya en registros que NO están…», «La vía
# alternativa se apoya…», el de cada problema y el «SIN APOYO». Al voltear, los
# apoyos de la alternativa pasan a la propuesta —la que se acepta de un botón—
# y el aviso seguía diciendo «la vía alternativa»: el secretario creía limpia
# la propuesta que llevaba el registro inventado. Se rehacen con lo que quedó,
# como ya se hacía tras la regla de la ejecutoria.
# ═══════════════════════════════════════════════════════════════════════════

_PREF_GLOBAL = ("La propuesta del asunto se apoya en registros que NO",
                "La vía alternativa se apoya en registros que NO")
_PREF_PROBLEMA = "La propuesta se apoya en registros que NO están en el acervo:"
_SIN_APOYO = "se propone SIN APOYO del acervo"


def rehacer_avisos_registros(avisos: list, glob, propuestas: list, material, i: int = -1,
                             apoyos_previos: list | None = None) -> None:
    """EN SITIO: quita los avisos de registros que se escribieron antes del
    volteo y los vuelve a calcular con la global y las propuestas de ahora.
    `i`/`apoyos_previos`: el principal y los apoyos que tenía antes (su aviso
    por problema se identifica por esos registros)."""
    try:
        import fase5_propuesta as _f5
    except Exception:
        return
    validos = {str(t.get("registro", "")) for t in (getattr(material, "tesis", None) or [])
               if isinstance(t, dict)}

    def _inventados(apoyos) -> list:
        out = []
        for a in apoyos or []:
            m = _f5._RX_CIFRA_REGISTRO.search(str(a))
            if m and m.group(1) not in validos:
                out.append(str(a))
        return out

    nuevos = [a for a in avisos if not str(a).startswith(_PREF_GLOBAL) and _SIN_APOYO not in str(a)]
    pral = propuestas[i] if 0 <= i < len(propuestas or []) else None
    if pral is not None:
        viejos = _inventados(apoyos_previos)
        if viejos:
            for k, a in enumerate(nuevos):
                if str(a).startswith(_PREF_PROBLEMA) and str(viejos) in str(a):
                    del nuevos[k]
                    break
        ahora = _inventados(_g(pral, "apoyos", None))
        if ahora and _g(pral, "alcanza", True):
            nuevos.append(f"{_PREF_PROBLEMA} {ahora}. No se citan hasta comprobarlos en el Semanario.")
    if _g(glob, "alcanza", False):
        try:
            nuevos.extend(_f5.revisar_global(glob, material))
        except Exception:
            pass
    nuevos.extend(
        f"«{_g(p, 'sentido', '')}» {_SIN_APOYO}. Una propuesta sin fundamento es una opinión: "
        f"compruébala antes de aceptarla."
        for p in propuestas or [] if _g(p, "alcanza", False) and not (_g(p, "apoyos", None) or []))
    avisos[:] = nuevos
