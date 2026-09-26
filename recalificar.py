# -*- coding: utf-8 -*-
"""RECALIFICAR CON LA PREMISA DEL CAMBIO DE SENTIDO (26-sep-2026).

LA DECISIÓN DE DAVID, literal: «si cambio sentido hay que tumbar y regenerar
con la premisa del cambio de sentido». Contesta la duda (a) del árbol de
decisión: cuando el secretario resuelve el principal en la vía CONTRARIA a la
que propuso el motor, los accesorios relacionados que él no tocó conservaban
la calificación que el motor les había escrito para la otra vía. Medido en el
banco Kingston: en 5 de 16 accesorios era «fundado», y el engrose real los negó
en los 5. Hasta hoy sólo se avisaba.

LAS DOS MITADES
  · TUMBAR (`arbol_decision.aplicar`, determinista): esa calificación no se usa
    y no se le enseña al modelo que regenera —sería un ancla—.
  · REGENERAR (aquí): el motor vuelve a calificar SÓLO esos accesorios, con la
    PREMISA del cambio —el principal, el sentido que dictó el secretario y SU
    razón— como hechos dados que no se discuten.

LO QUE NO HACE, Y NO SE NEGOCIA
  · Nunca toca lo que el secretario marcó ni su principal: sólo recalifica lo
    que él no tocó (la máquina nunca cambia el sentido que él dictó).
  · No ve la calificación tumbada. Recibe la pregunta del accesorio, lo que la
    fase 3 dice que combate, sus argumentos del inventario (con su cita) y el
    material del acervo que ya ve el estudio.
  · En el prompt van DESCRIPCIONES y DATOS, nunca frases modelo (un ejemplo
    escrito en el prompt se firma literal: medido tres veces en este proyecto).
  · La caída con el principal no se cree: si el motor dice que el accesorio
    PRESUPONE la premisa desestimada, la cita tiene que constar en su
    planteamiento, con la MISMA verificación que `arbol_decision.presupuesto`;
    si no consta, se descarta la caída y queda su calificación de fondo.

EL ESTADO (gunicorn -w 2: nada en memoria). En la columna `taller_sesiones.plan`
—la del plan, sin migración nueva—, rama `recalificaciones` = {clave: casilla}
y `recalificaciones_corridas`, escritas por el MISMO compare-and-set por `rev`
que usa el plan (`main._taller_plan_cas`). Como mucho seis casillas y seis
corridas por adelanto: el documento es por adelanto (`plan_estudio._doc`), y
rehacer el adelanto empieza de cero.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import re
import time
import unicodedata

# Sube cuando cambie el prompt, el catálogo o la validación: una recalificación
# hecha con otra versión no se reutiliza (entra en la clave).
VERSION = "recal-1"

# EL MODELO DE LAS FASES, con razonamiento MEDIO (contrato): el mismo que el
# planificador. Se lee al llamar, no al importar, para que una prueba lo cambie.
ESFUERZO = os.getenv("ESFUERZO_RECALIFICAR", "medium")
TOPE_SALIDA = int(os.getenv("RECALIFICAR_TOPE_SALIDA", "12000"))
# LO QUE SE ESPERA COMO MUCHO, los dos intentos juntos. Si vence, los
# pendientes quedan SIN CALIFICAR y el estudio los desarrolla con el material.
TOPE_S = float(os.getenv("RECALIFICAR_TOPE_S", "90"))
# Seis corridas por adelanto, contando todas (pantalla y gemelos), y seis
# casillas: la pantalla pide una por cada razón estable del principal.
TOPE_CORRIDAS = int(os.getenv("RECALIFICAR_TOPE_CORRIDAS", "6"))
MAX_CASILLAS = 6
# Una corrida «en curso» sin latido en este tiempo es de un worker que murió.
# La corrida no pasa de 90 s y late cada 30.
ABANDONADA_S = 120.0
LATIDO_S = 30.0

MAX_SEGMENTOS = 8
MAX_TESIS = 5
MAX_NORMAS = 8

# Qué es cada calificación, para el prompt: DESCRIPCIONES, no frases que copiar.
_DESCRIBE = {
    "fundado": "tiene razón y prospera",
    "esencialmente_fundado": "tiene razón en lo sustancial y prospera",
    "sustancialmente_fundado": "tiene razón en lo sustancial y prospera",
    "parcialmente_fundado": "tiene razón en una parte, que prospera",
    "fundado_insuficiente": "tiene razón pero no cambia el resultado: subsiste otra "
                            "consideración que sostiene lo resuelto",
    "infundado": "se examina en el fondo y no tiene razón",
    "inoperante": "no puede prosperar por cómo está planteado (no combate la "
                  "consideración que sostiene lo resuelto, es novedoso, parte de un "
                  "hecho que no es cierto, o no se preparó la violación)",
    "inatendible": "no puede atenderse por cómo o cuándo se planteó",
    "ineficaz": "aun con razón en lo que dice, no produce ningún efecto útil",
    "innecesario": "la premisa del secretario deja sin materia su estudio",
}


# ═══ UTILIDADES ═════════════════════════════════════════════════════════════

def _ws(t) -> str:
    return " ".join(str(t or "").split())


def _sin_acentos(t: str) -> str:
    t = unicodedata.normalize("NFD", str(t or ""))
    return "".join(ch for ch in t if unicodedata.category(ch) != "Mn")


def _clave_texto(x) -> str:
    return re.sub(r"[^a-z0-9 ]", "", _sin_acentos(_ws(x)).lower()).strip()


def _norm_sentido(s) -> str:
    return str(s or "").strip().lower().replace(" ", "_")


def _get(x, k, d=""):
    if isinstance(x, dict):
        return x.get(k, d)
    return getattr(x, k, d)


def _prospera(s: str) -> bool:
    try:
        import tipos_asunto as _ta
        return _ta.prospera(s)
    except Exception:                                   # pragma: no cover
        s = (s or "").lower()
        return "fundad" in s and "insuficien" not in s


# ═══ QUIÉN, Y CON QUÉ CLAVE ═════════════════════════════════════════════════

def pendientes(detalle: dict) -> list[str]:
    """Los problemas que `arbol_decision.aplicar` tumbó y siguen sin su
    calificación nueva (`detalle[t]["recalificar"]`)."""
    return [t for t, d in (detalle or {}).items()
            if isinstance(d, dict) and d.get("recalificar")]


def tumbados(detalle: dict) -> list[str]:
    """Todos los que el árbol tumbó en esta vuelta, recalificados o no."""
    return [t for t, d in (detalle or {}).items()
            if isinstance(d, dict) and d.get("de") in ("por_recalificar", "recalificada")]


def clave_de(detalle: dict) -> str:
    """La clave que el árbol calculó para esta recalificación ('' si no hay)."""
    for d in (detalle or {}).values():
        if isinstance(d, dict) and d.get("clave_recalificar"):
            return str(d["clave_recalificar"])
    return ""


def clave(principal_problema, sentido, razon, pendientes, huella_adelanto,
          tipo_asunto) -> str:
    """LA CLAVE DE UNA RECALIFICACIÓN: el principal, el sentido que fijó el
    secretario y SU razón literal (la premisa), qué accesorios se recalifican,
    el adelanto y el tipo de asunto. Con cualquiera de ellos distinto, la
    recalificación guardada no sirve. Los pendientes se ordenan: el mismo
    conjunto da la misma clave venga de la pantalla o de los gemelos.

    La razón que el secretario tecleó para un ACCESORIO sin elegir sentido NO
    entra: la pantalla la devuelve después junto con el sentido recalificado,
    y la clave tiene que ser la misma en las dos vueltas. Viaja guardada con
    el resultado (`razon_suya`) y el árbol la vuelve a poner."""
    # LOS TEXTOS, RECORTADOS COMO LOS RECORTA EL CRITERIO ARMADO (400): la
    # pantalla (/taller/reparto) manda el problema entero y los gemelos lo
    # traen recortado; la clave tiene que ser la misma por las dos puertas.
    _c = _corte()
    base = {"v": VERSION,
            "p": _clave_texto(str(principal_problema or "")[:_c]),
            "s": _norm_sentido(sentido),
            "r": _ws(razon),
            "pend": sorted(_clave_texto(str(x or "")[:_c]) for x in (pendientes or [])),
            "h": str(huella_adelanto or ""),
            "t": str(tipo_asunto or "").strip().lower()}
    return hashlib.sha1(json.dumps(base, ensure_ascii=False, sort_keys=True)
                        .encode()).hexdigest()[:20]


def clave_premisa(principal_problema, sentido, razon, huella_adelanto, tipo_asunto) -> str:
    """LA PREMISA SOLA: la clave sin los pendientes. Sirve para no rehacer lo
    que ya se recalificó cuando el secretario PISA uno de los tumbados
    (revisión adversarial, 26-sep-2026): el conjunto de pendientes cambia —y
    con él la clave—, pero la premisa es la misma y lo que el motor calificó
    para los demás sigue siendo de esa premisa. Sin esto, cada pastilla que él
    marcaba lanzaba otra corrida (de las seis del adelanto) y los demás podían
    salir distintos: «lo que el secretario marque después la sustituye», a
    ella, no a sus vecinos."""
    return "p-" + clave(principal_problema, sentido, razon, ["\x00premisa"], huella_adelanto,
                        tipo_asunto)


def premisa_de(detalle: dict) -> str:
    """La clave de la premisa que el árbol calculó ('' si no hay)."""
    for d in (detalle or {}).values():
        if isinstance(d, dict) and d.get("premisa_recalificar"):
            return str(d["premisa_recalificar"])
    return ""


def _cubre(resultados: dict, pendientes: list) -> bool:
    _c = _corte()
    tiene = {_clave_texto(str(x)[:_c]) for x, v in (resultados or {}).items() if isinstance(v, dict)}
    return bool(pendientes) and all(_clave_texto(str(x)[:_c]) in tiene for x in pendientes)


def casilla_de(recalificadas, k: str, premisa: str = "", pendientes: list = None):
    """La recalificación guardada con la clave `k`, o None. Acepta una sola
    ({"clave", "resultados"}) o la rama entera de la fila ({clave: casilla}).

    Sin la de la misma clave, y con la rama entera: una «listo» de la MISMA
    premisa (`premisa`, ver `clave_premisa`) que ya calificó todos los
    `pendientes` —la de antes de que él pisara alguno—."""
    if not isinstance(recalificadas, dict) or not k:
        return None
    if recalificadas.get("clave") == k and isinstance(recalificadas.get("resultados"), dict):
        return recalificadas
    c = recalificadas.get(k)
    if isinstance(c, dict) and isinstance(c.get("resultados"), dict) and c.get("resultados"):
        return c
    if not premisa or not pendientes:
        return None
    mejores = [x for x in recalificadas.values()
               if isinstance(x, dict) and x.get("premisa") == premisa
               and x.get("estado") == "listo" and isinstance(x.get("resultados"), dict)
               and _cubre(x["resultados"], pendientes)]
    if not mejores:
        return None
    return max(mejores, key=lambda x: float(x.get("hecho") or 0))


def catalogo(via_prospera: bool, procesal: bool = False) -> tuple:
    """Las calificaciones admitidas: las del árbol, sin «innecesario» ni «sin
    materia» —salvo que la premisa deje sin materia: sólo en la vía en que el
    principal PROSPERA y nunca en una violación procesal (arts. 74, fr. V, y
    174 de la Ley de Amparo: se deciden todas; la única salida es el 189, que
    ya aplica el árbol antes de llegar aquí)—."""
    import arbol_decision as _ad
    fuera = [s for s in _ad._SENTIDOS if s not in ("sin_materia", _ad.INNECESARIO)]
    if via_prospera and not procesal:
        fuera.append(_ad.INNECESARIO)
    return tuple(fuera)


# ═══ LO QUE VE EL MODELO ════════════════════════════════════════════════════

def _corte() -> int:
    try:
        import arbol_decision as _ad
        return int(_ad.CORTE_PROBLEMA)
    except Exception:                                   # pragma: no cover
        return 400


def _fase3(r, problema: str):
    """(número 1..n, dict de la fase 3) del problema, por su pregunta. El
    criterio armado recorta el problema a 400 caracteres: se casa también por
    esos primeros caracteres (sin eso, un planteamiento largo llegaba al
    modelo sin número y sin lo que combate)."""
    probs = list(getattr(getattr(r, "fases", None), "problemas", None) or [])
    k = _clave_texto(problema)
    _c = _corte()
    k_c = _clave_texto(str(problema or "")[:_c])
    for i, p in enumerate(probs, 1):
        d = p if isinstance(p, dict) else {"pregunta": str(p)}
        for campo in ("pregunta", "pregunta_original"):
            v = str(d.get(campo) or "")
            if v and (_clave_texto(v) == k or _clave_texto(v[:_c]) == k_c):
                return i, d
    return 0, {"pregunta": problema}


def _escrito(fases) -> str:
    f = list(_get(fases, "fuentes", None) or []) + ["", ""]
    return str(f[1] or "")


def _parrafos_resumen(fases) -> list[str]:
    return [p.strip() for p in str(_get(fases, "resumen_conceptos", "") or "").split("\n")
            if p.strip()]


def _inventario(r) -> list:
    """Los segmentos del inventario (una vez por prompt), o [] sin él."""
    fases = getattr(r, "fases", None)
    e = getattr(r, "encargo", None)
    try:
        import inventario as _inv
        return [s for s in (_inv.segmentos(fases, _escrito(fases),
                                           bool(getattr(e, "es_recurso", False))) or [])
                if isinstance(s, dict)]
    except Exception:
        return []


def argumentos_de(r, numero: int, fase3: dict, segs: list = None) -> list[dict]:
    """Los argumentos del escrito que corresponden a ese planteamiento: los
    segmentos del inventario cuyo concepto está en su «cubre» (con dos o más
    planteamientos contados); si no se puede decir por el «cubre», los que
    comparten más anclas con su pregunta y lo que combate. Sin inventario, los
    párrafos del resumen con esas anclas. Cada uno con su cita literal."""
    import arbol_decision as _ad
    fases = getattr(r, "fases", None)
    if segs is None:
        segs = _inventario(r)
    try:
        import fases123_pipeline as _f123
        cubre = _f123.cubre_de(fase3)
    except Exception:
        cubre = []
    conteo = _get(fases, "conteo", None) or {}
    try:
        n = int(conteo.get("n") or 0) if str(conteo.get("estado")) == "contado" else 0
    except (TypeError, ValueError, AttributeError):
        n = 0
    if segs and cubre and n >= 2:
        sel = [s for s in segs if int(s.get("concepto") or 0) in cubre]
        if sel:
            return [{"id": str(s.get("id") or ""), "texto": _ws(s.get("texto"))[:600],
                     "cita": _ws(s.get("cita"))[:400]} for s in sel[:MAX_SEGMENTOS]]
    ancla = _ad._anclas(f"{fase3.get('pregunta', '')} {fase3.get('combate', '')}")
    if segs:
        rank = sorted(((len(_ad._anclas(s.get("texto")) & ancla), i, s) for i, s in enumerate(segs)),
                      key=lambda x: (-x[0], x[1]))
        sel = [s for k, _, s in rank if k >= 2][:MAX_SEGMENTOS // 2]
        return [{"id": str(s.get("id") or ""), "texto": _ws(s.get("texto"))[:600],
                 "cita": _ws(s.get("cita"))[:400]} for s in sel]
    pars = _parrafos_resumen(fases)
    rank = sorted(((len(_ad._anclas(p) & ancla), i, p) for i, p in enumerate(pars)),
                  key=lambda x: (-x[0], x[1]))
    return [{"id": "", "texto": _ws(p)[:900], "cita": ""} for k, _, p in rank if k >= 2][:3]


def _indice(material, fases) -> dict:
    """El material TAL COMO LO VE EL ESTUDIO (el índice del plan: el mismo
    recorte de tesis y de preceptos) y, de cada tesis, para qué problema se
    buscó («para»)."""
    if material is None:
        return {"tesis": [], "normas": []}
    try:
        import plan_estudio as _pe
        ind = _pe.indice_material(material, fases)
    except Exception:
        ind = {"tesis": [{"registro": str(t.get("registro") or ""), "rubro": _ws(t.get("rubro"))[:400],
                          "obligatoria": bool(t.get("obligatoria")), "texto": str(t.get("texto") or "")}
                         for t in (getattr(material, "tesis", None) or [])[:10] if isinstance(t, dict)],
               "normas": [{"cuerpo_legal": _ws(n.get("cuerpo_legal"))[:200], "articulo": _ws(n.get("articulo")),
                           "texto": str(n.get("texto") or "")}
                          for n in (getattr(material, "normas", None) or [])[:12] if isinstance(n, dict)]}
    para = {}
    for t in (getattr(material, "tesis", None) or []):
        if isinstance(t, dict) and t.get("registro"):
            para[str(t["registro"]).strip()] = [int(x) for x in (t.get("para") or [])
                                                if str(x).isdigit()]
    for t in ind.get("tesis") or []:
        t["para"] = para.get(str(t.get("registro") or "").strip(), [])
    return ind


def _tesis_para(ind: dict, numero: int) -> list[dict]:
    tesis = ind.get("tesis") or []
    suyas = [t for t in tesis if numero and numero in (t.get("para") or [])]
    return (suyas or tesis[:3])[:MAX_TESIS]


def entradas(r, crit, detalle: dict) -> tuple:
    """(principal, accesorios) para `recalificar`, desde el criterio YA armado
    —con el árbol aplicado— y su `detalle`. El principal lleva el sentido y la
    razón que fijó el secretario; cada accesorio, su pregunta, su dict de la
    fase 3 y si es una violación procesal. Nunca la calificación tumbada."""
    pend = pendientes(detalle)
    p_txt = ""
    for t in pend:
        p_txt = str((detalle.get(t) or {}).get("principal") or "")
        if p_txt:
            break
    pc = next((c for c in (crit or []) if str(_get(c, "problema", "")) == p_txt), None)
    if pc is None:
        pc = next((c for c in (crit or []) if str(_get(c, "jerarquia", "")).lower() == "principal"),
                  None)
    n_p, f3_p = _fase3(r, str(_get(pc, "problema", "") or p_txt))
    principal = {"problema": str(_get(pc, "problema", "") or p_txt), "numero": n_p,
                 "sentido": _norm_sentido(_get(pc, "sentido", "")),
                 "razon": str(_get(pc, "razonamiento", "") or ""), "fase3": f3_p}
    acc = []
    por_c = {str(_get(c, "problema", "")): c for c in (crit or [])}
    _usados = {n_p}
    for t in pend:
        n, f3 = _fase3(r, t)
        if not n or n in _usados:
            # Sin su problema de la fase 3 no hay número: se le da uno que no
            # choque. La respuesta se casa por número, y dos «0» se llevaban
            # la misma calificación (revisión adversarial, 26-sep-2026).
            n = max([100] + list(_usados)) + 1
        _usados.add(n)
        d = detalle.get(t) or {}
        acc.append({"problema": t, "numero": n, "fase3": f3,
                    "procesal": bool(d.get("procesal")),
                    # La razón que tecleó él (el árbol sólo la deja en un tumbado
                    # cuando es suya): dato, no ancla de la otra vía.
                    "razon_suya": (str(_get(por_c.get(t), "razonamiento", "") or "")
                                   if d.get("razon_suya") else "")})
    return principal, acc


def prompt(r, material, principal: dict, accesorios: list, contexto: str = "",
           suplencia=None, faltas: list = None) -> str:
    """El prompt: DESCRIPCIONES y DATOS. Ninguna frase para copiar."""
    fases = getattr(r, "fases", None)
    e = getattr(r, "encargo", None)
    tipo = str(getattr(e, "tipo_asunto", "") or "amparo_directo").replace("_", " ")
    via_p = _prospera(principal.get("sentido", ""))
    f3p = principal.get("fase3") or {}
    ind = _indice(material, fases)
    L = [
        f"Recalificas planteamientos accesorios del proyecto de sentencia de un {tipo} "
        f"ante un tribunal colegiado de circuito. El secretario resolvió el problema "
        f"PRINCIPAL en la vía contraria a la que había propuesto el motor. Las "
        f"calificaciones que el motor había escrito para estos accesorios suponían el "
        f"principal resuelto al revés y se retiraron: no las conoces y no las "
        f"reconstruyas. Califica cada accesorio de nuevo, por lo que él mismo plantea, "
        f"con la PREMISA del secretario como hecho dado.",
        "",
        "LA PREMISA — la fijó el secretario; no se discute, no se matiza y no se vuelve a razonar:",
        f"  problema principal: {principal.get('problema', '')}",
    ]
    if f3p.get("resolvio"):
        L.append(f"  lo que resolvió el órgano sobre él: {_ws(f3p['resolvio'])[:900]}")
    if f3p.get("combate"):
        L.append(f"  lo que se combate en él: {_ws(f3p['combate'])[:900]}")
    L.append(f"  sentido que fijó el secretario: {principal.get('sentido', '').replace('_', ' ')} "
             f"({'el principal PROSPERA' if via_p else 'el principal NO prospera'})")
    L.append("  razón del secretario (literal): "
             + (_ws(principal.get("razon"))[:2500] or "no escribió razón; el sentido es la premisa"))
    acto = _ws(_get(fases, "resumen_acto", ""))
    if acto:
        L += ["", "LO QUE RESOLVIÓ EL ACTO (resumen):", acto[:3500]]
    L += ["", "LOS PLANTEAMIENTOS QUE SE RECALIFICAN:"]
    _segs = _inventario(r)
    for a in accesorios:
        f3 = a.get("fase3") or {}
        L.append(f"PLANTEAMIENTO {a['numero']} · clase: "
                 f"{'violación procesal' if a.get('procesal') else 'fondo'}")
        L.append(f"  pregunta: {a['problema']}")
        if f3.get("resolvio"):
            L.append(f"  lo que resolvió el órgano: {_ws(f3['resolvio'])[:900]}")
        L.append(f"  se combate diciendo: {_ws(f3.get('combate'))[:1500] or '(la fase 3 no lo resumió)'}")
        if _ws(a.get("razon_suya")):
            L.append("  razón que escribió el secretario para este planteamiento (literal; manda: "
                     "la calificación tiene que ser coherente con ella): "
                     + _ws(a["razon_suya"])[:1500])
        args = argumentos_de(r, a["numero"], f3, _segs)
        if args:
            L.append("  argumentos del escrito:")
            for s in args:
                L.append(f"    {s['id'] + ' · ' if s['id'] else ''}{s['texto']}"
                         + (f" · cita literal: «{s['cita']}»" if s.get("cita") else ""))
        ts = _tesis_para(ind, a["numero"])
        if ts:
            L.append("  tesis del acervo para este planteamiento:")
            for t in ts:
                L.append(f"    [registro {t.get('registro', '')}] "
                         f"{'obligatoria' if t.get('obligatoria') else 'orientadora'} · "
                         f"{t.get('rubro', '')}\n      {_ws(t.get('texto'))[:700]}")
        L.append("  calificaciones admitidas: "
                 + ", ".join(catalogo(via_p, bool(a.get("procesal")))))
        L.append("")
    normas = (ind.get("normas") or [])[:MAX_NORMAS]
    if normas:
        L.append("PRECEPTOS DEL ACERVO (los que verá el estudio):")
        for n in normas:
            L.append(f"  {n.get('cuerpo_legal', '')} — art. {n.get('articulo', '')}: "
                     f"{_ws(n.get('texto'))[:450]}")
        L.append("")
    if _ws(contexto):
        L += ["LO QUE APORTÓ EL SECRETARIO:", _ws(contexto)[:4000], ""]
    try:
        import suplencia as _sp
        if _sp.confirmada(suplencia):
            L += [f"SUPLENCIA DE LA QUEJA CONFIRMADA por el secretario: fracción "
                  f"{suplencia.get('fraccion')}, a favor de {_ws(suplencia.get('a_favor_de'))[:200]}. "
                  f"A esa parte no se le declara inoperante un planteamiento por cómo lo formuló.", ""]
    except Exception:
        pass
    usadas = sorted({s for a in accesorios for s in catalogo(via_p, bool(a.get("procesal")))},
                    key=lambda x: list(_DESCRIBE).index(x) if x in _DESCRIBE else 99)
    L += [
        "CÓMO SE CALIFICA (son descripciones: no copies ninguna expresión de aquí a la razón):",
        "- La premisa manda. Nada de lo que escribas puede suponer el principal resuelto en el "
        "otro sentido ni contradecir la razón del secretario.",
        "- sentido: una de las calificaciones admitidas de ESE planteamiento. Qué es cada una: "
        + "; ".join(f"{k} = {_DESCRIBE.get(k, '')}" for k in usadas) + ".",
        "- razon: de dos a cinco frases con la razón que decide ese planteamiento, con los datos "
        "del propio planteamiento y, si sirve, el registro de una tesis o el artículo de un "
        "precepto de los que están arriba. Ningún registro ni artículo que no esté arriba.",
    ]
    if not via_p:
        L.append(
            "- presupone: sólo cuando el argumento ENTERO del planteamiento sólo tiene sentido si "
            "la premisa que el secretario desestimó fuera cierta. Entonces un objeto con «cita» "
            "—seis palabras o más, LITERALES y seguidas, sin cortes ni puntos suspensivos, "
            "copiadas de «se combate diciendo» de ESE planteamiento, donde da por cierta esa "
            "premisa— y «por_que» —la premisa que da por cierta, en una frase—. Si además "
            "plantea algo por su cuenta, null. En una violación procesal, siempre null: se "
            "decide por lo que ella plantea (artículos 74, fracción V, y 174 de la Ley de "
            "Amparo). Aunque declares presupone, «sentido» y «razon» son la calificación de "
            "fondo que tendría si se estudiara.")
    else:
        L.append("- presupone: siempre null (el principal prospera).")
    if faltas:
        L += ["", "LO QUE FALLÓ EN TU RESPUESTA ANTERIOR (corrígelo):"] + [f"  · {x}" for x in faltas[:12]]
    L += ["", "Responde SÓLO con JSON:",
          '{"planteamientos": [{"numero": <número del planteamiento>, "sentido": "<una de sus '
          'calificaciones admitidas>", "razon": "<la razón>", "presupone": null | {"cita": '
          '"<literal>", "por_que": "<la premisa que da por cierta>"}}]}']
    return "\n".join(L)


# ═══ LA LLAMADA ═════════════════════════════════════════════════════════════

def _modelo() -> str:
    m = os.getenv("MODELO_RECALIFICAR", "").strip()
    if m:
        return m
    import fases123_pipeline as _f123
    return _f123.MODELO_FASES


async def _llamar(cliente, texto: str) -> str:
    """Una llamada con JSON estricto por `llamada_modelo.crear`, con el
    razonamiento MEDIO. Si vuelve vacía o cortada, otra con el doble de cupo:
    el razonamiento se MANTIENE."""
    import llamada_modelo as _lm
    kw = dict(model=_modelo(), messages=[{"role": "user", "content": texto}],
              max_completion_tokens=TOPE_SALIDA, response_format={"type": "json_object"})
    if ESFUERZO:
        kw["reasoning_effort"] = ESFUERZO
    r = await _lm.crear(cliente, **kw)
    txt = (r.choices[0].message.content or "").strip()
    fin = str(getattr(r.choices[0], "finish_reason", "") or "").lower()
    if not txt or fin == "length":
        kw["max_completion_tokens"] = TOPE_SALIDA * 2
        r = await _lm.crear(cliente, **kw)
        txt = (r.choices[0].message.content or "").strip()
    return txt


def _json_de(txt: str) -> dict:
    try:
        d = json.loads(txt)
        return d if isinstance(d, dict) else {}
    except Exception:
        m = re.search(r"\{.*\}", str(txt or ""), re.S)
        if m:
            try:
                d = json.loads(m.group(0))
                return d if isinstance(d, dict) else {}
            except Exception:
                return {}
    return {}


# ═══ LA VALIDACIÓN ══════════════════════════════════════════════════════════

def validar(crudo: dict, accesorios: list, principal: dict) -> tuple:
    """(resultados, faltas, avisos). `resultados` = {problema: {sentido, razon,
    presupone, verificado}} de los que pasan; `faltas` = lo que obliga a
    pedirlo otra vez (sentido fuera del catálogo, razón vacía, planteamiento
    ausente); `avisos` = {problema: texto} de la caída que no consta —no es
    falta: se descarta la caída y queda su calificación de fondo—."""
    import arbol_decision as _ad
    via_p = _prospera(principal.get("sentido", ""))
    items = crudo.get("planteamientos") if isinstance(crudo, dict) else None
    por_num: dict = {}
    for it in (items if isinstance(items, list) else []):
        if not isinstance(it, dict):
            continue
        try:
            por_num.setdefault(int(it.get("numero")), it)
        except (TypeError, ValueError):
            continue
    resultados, faltas, avisos = {}, [], {}
    for a in accesorios:
        t = a["problema"]
        it = por_num.get(int(a.get("numero") or 0))
        if it is None:
            faltas.append(f"falta el planteamiento {a.get('numero')}")
            continue
        s = _ad._sentido_valido(it.get("sentido"))
        cat = catalogo(via_p, bool(a.get("procesal")))
        if s not in cat:
            faltas.append(f"planteamiento {a.get('numero')}: «{_ws(it.get('sentido'))[:40]}» no es una "
                          f"de sus calificaciones admitidas ({', '.join(cat)})")
            continue
        razon = _ws(it.get("razon"))
        if len(razon.split()) < 5:
            faltas.append(f"planteamiento {a.get('numero')}: la razón está vacía o no dice nada")
            continue
        pre = it.get("presupone")
        if isinstance(pre, str) and not _ad._vacio(pre):
            pre = {"cita": pre}
        presupone, verificado = None, False
        if isinstance(pre, dict) and not _ad._vacio(pre.get("cita")):
            if via_p or a.get("procesal"):
                avisos[t] = (f"«{t[:90]}»: el motor dijo que presupone la premisa, pero "
                             + ("el principal prospera" if via_p else "es una violación procesal, "
                                "que se decide por lo que plantea")
                             + f"; se queda con su calificación de fondo ({s.replace('_', ' ')})")
            else:
                ev = _ad.presupuesto(
                    {"presupone": {"cita": pre.get("cita"), "premisa": pre.get("por_que") or "",
                                   "causa_propia": None}},
                    a.get("fase3") or t, principal.get("fase3") or principal.get("problema", ""))
                presupone = {"cita": ev["cita"][:600], "por_que": _ws(pre.get("por_que"))[:400]}
                verificado = bool(ev["verificado"])
                if not verificado:
                    avisos[t] = (f"«{t[:90]}»: el motor dijo que cae con el principal, pero la cita "
                                 f"no consta en su planteamiento ({ev['motivo'] or 'sin cita'}); se "
                                 f"descarta la caída y queda su calificación de fondo "
                                 f"({s.replace('_', ' ')})")
        resultados[t] = {"sentido": s, "razon": razon[:1500], "presupone": presupone,
                         "verificado": verificado}
    return resultados, faltas, avisos


# ═══ LA RECALIFICACIÓN ══════════════════════════════════════════════════════

async def recalificar(r, material, principal: dict, accesorios: list[dict], contexto: str = "",
                      suplencia=None, cliente=None, *, clave_: str = "",
                      tope_s: float = None) -> dict:
    """UNA llamada para todos los pendientes del asunto y un reintento si el
    JSON no pasa la validación; todo dentro de `tope_s` (90 s). Nunca lanza.

    → {"clave", "estado": "listo"|"fallo"|"error", "resultados": {problema:
       {"sentido", "razon", "presupone": {"cita", "por_que"}|None,
        "verificado": bool}}, "avisos": [...], "segundos", "intentos"}
    «listo»: todos calificados. «fallo»: la validación no pasó dos veces (lo
    que pasó se guarda; lo demás queda SIN CALIFICAR). «error»: el proveedor o
    el tope de tiempo; la siguiente petición lo reintenta."""
    t0 = time.time()
    tope = TOPE_S if tope_s is None else float(tope_s)
    e = getattr(r, "encargo", None)
    if not clave_:
        try:
            import taller_estado as _te
            _h = _te.huella_contraste(r)
        except Exception:
            _h = ""
        clave_ = clave(principal.get("problema", ""), principal.get("sentido", ""),
                       principal.get("razon", ""), [a["problema"] for a in accesorios],
                       _h, str(getattr(e, "tipo_asunto", "") or ""))
    acc = [a for a in accesorios if a.get("problema")]
    if not acc:
        return {"clave": clave_, "estado": "listo", "resultados": {}, "avisos": [],
                "segundos": 0.0, "intentos": 0}
    mejor: dict = {}
    avisos_de: dict = {}
    estado_int = {"intentos": 0, "faltas": []}

    async def _dos():
        faltas: list = []
        for _ in range(2):
            estado_int["intentos"] += 1
            faltan = [a for a in acc if a["problema"] not in mejor]
            txt = await _llamar(cliente, prompt(r, material, principal, faltan, contexto,
                                                suplencia, faltas))
            res, faltas, av = validar(_json_de(txt), faltan, principal)
            mejor.update(res)
            avisos_de.update(av)
            estado_int["faltas"] = list(faltas)
            if all(a["problema"] in mejor for a in acc):
                return

    estado = "listo"
    avisos: list = []
    try:
        await asyncio.wait_for(_dos(), timeout=tope)
        if not all(a["problema"] in mejor for a in acc):
            estado = "fallo"
            avisos.append("la recalificación no pasó la validación dos veces: "
                          + "; ".join(estado_int["faltas"][:3]))
    except asyncio.TimeoutError:
        estado = "error"
        avisos.append(f"la recalificación no llegó en {tope:.0f} s")
    except Exception as ex:
        estado = "error"
        avisos.append(f"la recalificación falló ({type(ex).__name__})")
    if estado != "listo" and all(a["problema"] in mejor for a in acc):
        estado = "listo"
    # LA RAZÓN QUE TECLEÓ ÉL viaja con el resultado: el árbol la vuelve a poner
    # cuando la pantalla devuelva el sentido recalificado (ver `clave`).
    for a in acc:
        if a["problema"] in mejor and _ws(a.get("razon_suya")):
            mejor[a["problema"]]["razon_suya"] = _ws(a["razon_suya"])[:2000]
    avisos = [avisos_de[t] for t in mejor if t in avisos_de] + avisos
    return {"clave": clave_, "estado": estado, "resultados": dict(mejor),
            "avisos": avisos[:20], "segundos": round(time.time() - t0, 1),
            "intentos": estado_int["intentos"]}


# ═══ EL ESTADO EN LA FILA (taller_sesiones.plan), EN FUNCIONES PURAS ═════════
# Las lee y escribe `main._taller_plan_cas`, el compare-and-set del plan: el
# documento es UNO por fila, y la rama del plan y ésta conviven en él. Se parte
# de `plan_estudio._doc`, que lo rehace si es de otro adelanto (el plan y las
# recalificaciones de otro adelanto no sirven).

def _base(doc, huella: str) -> dict:
    import plan_estudio as _pe
    d = _pe._doc(doc, huella)
    if not isinstance(d.get("recalificaciones"), dict):
        d["recalificaciones"] = {}
    d["recalificaciones_corridas"] = int(d.get("recalificaciones_corridas") or 0)
    return d


def abandonada(c: dict, ahora: float) -> bool:
    if not isinstance(c, dict) or c.get("estado") != "en_curso":
        return False
    try:
        ultimo = float(c.get("latido") or c.get("desde") or 0)
    except (TypeError, ValueError):
        ultimo = 0.0
    return ahora - ultimo > ABANDONADA_S


def fila_pedir(doc, k: str, huella: str, ahora: float, tope: int = TOPE_CORRIDAS) -> tuple:
    """(doc nuevo | None, decisión): «listo» · «fallo» (ya se calculó: no se
    recalcula) · «en_curso» (se espera, no se duplica) · «tope» · «lanzar» (se
    reservó la corrida: quien llama la corre). Un «error» del proveedor se
    vuelve a lanzar, y cuenta."""
    d = _base(doc, huella)
    c = d["recalificaciones"].get(k)
    if isinstance(c, dict):
        if c.get("estado") in ("listo", "fallo"):
            return None, c["estado"]
        if c.get("estado") == "en_curso" and not abandonada(c, ahora):
            return None, "en_curso"
    if d["recalificaciones_corridas"] >= tope:
        return None, "tope"
    d["recalificaciones_corridas"] += 1
    d["recalificaciones"][k] = {"estado": "en_curso", "desde": ahora, "latido": ahora}
    _podar(d["recalificaciones"], k)
    return d, "lanzar"


def fila_latido(doc, k: str, huella: str, ahora: float) -> tuple:
    if not isinstance(doc, dict) or doc.get("huella") != huella:
        return None, "nada"
    d = _base(doc, huella)
    c = d["recalificaciones"].get(k)
    if not isinstance(c, dict) or c.get("estado") != "en_curso":
        return None, "nada"
    c["latido"] = ahora
    return d, "latido"


def fila_resultado(doc, k: str, huella: str, salida: dict, segundos: float,
                   ahora: float) -> tuple:
    """Guarda el resultado en SU casilla. Si la fila ya es de otro adelanto,
    no escribe nada."""
    if not isinstance(doc, dict) or doc.get("huella") != huella:
        return None, "otro_adelanto"
    d = _base(doc, huella)
    est = str((salida or {}).get("estado") or "error")
    d["recalificaciones"][k] = {
        "estado": est if est in ("listo", "fallo", "error") else "error",
        "resultados": dict((salida or {}).get("resultados") or {}),
        "avisos": list((salida or {}).get("avisos") or [])[:20],
        "segundos": round(float(segundos or 0), 1), "hecho": ahora,
        # La premisa sola: lo que sirve cuando él pisa uno (`casilla_de`).
        "premisa": str((salida or {}).get("premisa") or "")}
    _podar(d["recalificaciones"], k)
    return d, est


def _podar(casillas: dict, conservar: str) -> None:
    if len(casillas) <= MAX_CASILLAS:
        return
    viejas = sorted((x for x in casillas if x != conservar),
                    key=lambda x: float((casillas[x] or {}).get("hecho")
                                        or (casillas[x] or {}).get("desde") or 0))
    for x in viejas[:len(casillas) - MAX_CASILLAS]:
        casillas.pop(x, None)


def guardadas(doc, huella: str) -> dict:
    """{clave: casilla} de las recalificaciones terminadas de ESTE adelanto que
    traen resultados (las que `arbol_decision.aplicar` sabe aplicar)."""
    if not isinstance(doc, dict) or doc.get("huella") != huella:
        return {}
    return {k: copy.deepcopy(c) for k, c in (doc.get("recalificaciones") or {}).items()
            if isinstance(c, dict) and c.get("estado") in ("listo", "fallo", "error")
            and c.get("resultados")}


def agotadas(doc, huella: str, tope: int = TOPE_CORRIDAS) -> bool:
    """¿Se gastaron las corridas de este adelanto? Entonces nadie calculará
    una premisa nueva: quien espera no debe esperar."""
    if not isinstance(doc, dict) or doc.get("huella") != huella:
        return False
    try:
        return int(doc.get("recalificaciones_corridas") or 0) >= tope
    except (TypeError, ValueError):
        return False


def estado_de(doc, huella: str, k: str, ahora: float) -> dict:
    """{"estado": listo|en_curso|fallo|sin_calcular, "casilla"}. Una corrida
    sin latido, o un error del proveedor, se leen «fallo» (la pantalla deja de
    esperar; el pedido siguiente lo reintenta)."""
    if not isinstance(doc, dict) or doc.get("huella") != huella:
        return {"estado": "sin_calcular", "casilla": None}
    c = (doc.get("recalificaciones") or {}).get(k)
    if not isinstance(c, dict):
        return {"estado": "sin_calcular", "casilla": None}
    est = c.get("estado")
    if est == "en_curso" and abandonada(c, ahora):
        est = "fallo"
    if est == "error":
        est = "fallo"
    return {"estado": est if est in ("listo", "en_curso", "fallo") else "sin_calcular",
            "casilla": copy.deepcopy(c)}


# ═══ LO QUE SE LE DICE AL SECRETARIO ════════════════════════════════════════

def aviso_sin_calificar(problemas: list, sentido_principal: str) -> str:
    """El aviso cuando la recalificación no llegó: los tumbados quedan SIN
    CALIFICAR. Va en los avisos del proyecto en TODAS las variantes (la v1,
    congelada, sólo recibe esto)."""
    return ("SIN CALIFICAR TRAS TU CAMBIO DE SENTIDO: "
            + " · ".join(f"«{t[:80]}»" for t in problemas[:6])
            + f". Con el principal {str(sentido_principal or '').replace('_', ' ')} —la vía "
              "contraria a la que propuso el motor— su calificación de la otra vía se retiró y la "
              "recalificación con tu premisa no llegó. El estudio los desarrolla con el material y "
              "los pone PRIMERO en ADVERTENCIAS; califícalos tú si quieres otra cosa.")
