# -*- coding: utf-8 -*-
"""LA FUENTE QUE LLEGA DESPUÉS DE REDACTAR (rediseño del taller, punto 7,
29-sep-2026).

POR QUÉ. David: «redactor_adelanto.resolver y su camino en vivo pueden recuperar
tesis y preceptos después de redactar. Esa recuperación permite completar citas
y notas, pero no implica que el razonamiento escrito antes haya sido contrastado
con la nueva fuente. Cuando una fuente incorporada después de redactar afecta
una premisa sustantiva, se revisan esa premisa, sus conclusiones y sus efectos
antes de componer el proyecto.»

Hoy (medido en el código, 29-sep): lo que entra tarde —las tesis que el estudio
o la parte citan por registro (`completar_tesis_citadas`), los preceptos que el
estudio cita sin tenerlos (`completar_preceptos`)— produce ficha, nota al pie y
avisos; nunca reabre nada.

QUÉ HACE ESTE MÓDULO (sin modelo, puro, probado aparte):
  · `clasificar(estudio, tesis_nuevas, normas_nuevas)` dice, por cada fuente
    tardía, en qué párrafos del estudio se usa, a qué UNIDADES pertenecen esos
    párrafos (las marcas del plan: C/A/AD/S son los argumentos que se
    contestan, M las premisas, U los efectos) y si el uso es SUSTANTIVO —el
    párrafo contesta un argumento o expone una premisa o un efecto— o no.
  · `informe_y_aviso` arma el aviso para el secretario y el estado de salida:
    si alguna fuente tardía toca una unidad sustantiva, el proyecto sale como
    «justificacion_pendiente» (punto 8), nombrando las unidades, en vez de
    pasar por verificado.

LO QUE TODAVÍA NO HACE, Y POR QUÉ. Reabrir la unidad con el modelo —reescribir
esa premisa y sus dependientes con el texto de la fuente— es una llamada del
modelo del estudio por unidad (hasta 90 s cada una) y depende de decisiones que
son de David: qué pasa si la fuente contradice el sentido que él dictó, cuántas
unidades se reabren como máximo, y si un precepto que sólo existe en internet
basta para reabrir. Mientras tanto, NO se finge: la unidad afectada se nombra
y el proyecto no sale como verificado.

SEÑAL, NO BLOQUEO. David, punto 8: «No activaría indiscriminadamente como
bloqueantes todas las heurísticas en sombra». Aquí la decisión «sustantiva» sale
de marcas que el propio estudio escribió (qué párrafo contesta qué), no de
adivinar el texto; lo único textual es reconocer la cita, y un uso que no se
reconoce se reporta «sin_unidad», no se inventa.
"""
from __future__ import annotations

import re
import unicodedata

SUSTANTIVAS = ("C", "A", "AD", "S", "M", "U")


def _plano(t) -> str:
    t = unicodedata.normalize("NFD", str(t or "").lower())
    return "".join(ch for ch in t if unicodedata.category(ch) != "Mn")


def _prefijo(i: str) -> str:
    m = re.match(r"^(AD|[CASMU])", str(i or "").upper())
    return m.group(1) if m else ""


def parrafos_con_unidades(estudio: str) -> list:
    """[(texto del párrafo, [ids de su unidad])]. Una marca sola en su renglón
    vale para el párrafo siguiente (la misma regla de `marcas.separar_marcas`)."""
    try:
        import marcas as _mc
    except Exception:                                   # pragma: no cover
        _mc = None
    fuera, pendientes = [], []
    for renglon in (estudio or "").split("\n"):
        if not renglon.strip():
            continue
        ids = []
        limpio = renglon
        if _mc is not None:
            for ini, fin, ii in _mc.marcas_en(renglon):
                ids.extend(ii)
            limpio = _mc.sin_marcas(renglon) if ids else renglon
        if ids and not limpio.strip():
            pendientes.extend(ids)          # marca sola: vale para el siguiente
            continue
        fuera.append((limpio, list(dict.fromkeys(pendientes + ids))))
        pendientes = []
    return fuera


def _cita_tesis(parrafo: str, t: dict) -> bool:
    reg = str((t or {}).get("registro") or "").strip()
    if reg and re.search(r"(?<!\d)" + re.escape(reg) + r"(?!\d)", parrafo):
        return True
    try:
        import fuerza_juridica as _fj
        clave = _fj.clave_de_tesis(t)
    except Exception:                                   # pragma: no cover
        clave = ""
    clave = (clave or "").replace(" ", "")
    return bool(clave) and clave in parrafo.replace(" ", "")


def _palabras_ley(nombre: str) -> set:
    vacias = {"de", "del", "la", "las", "los", "el", "y", "para", "en", "ley", "codigo", "federal"}
    return {w for w in re.findall(r"[a-z]{4,}", _plano(nombre)) if w not in vacias}


def _cita_norma(parrafo: str, n: dict) -> bool:
    art = str((n or {}).get("articulo") or "").strip()
    if not art:
        return False
    p = _plano(parrafo)
    if not re.search(r"\bart(?:iculos?|s?\.)[^.;]{0,60}?(?<!\d)" + re.escape(_plano(art)) + r"(?!\d)", p):
        return False
    ley = str(n.get("cuerpo_legal") or n.get("fuente") or "")
    pal = _palabras_ley(ley)
    # Sin nombre de ley legible, el número solo no basta para afirmar la cita.
    return bool(pal) and any(w in p for w in pal)


def clasificar(estudio: str, tesis_nuevas=(), normas_nuevas=(), *, mapa: dict = None,
               parrafos: list = None) -> list:
    """Por cada fuente tardía: {fuente, clase, unidades, sustantiva, parrafos,
    vincula?, de_internet?}. `sustantiva` es True si algún párrafo que la usa
    pertenece a una unidad que contesta un argumento, expone una premisa o
    fija un efecto.

    Con el estudio todavía marcado, las unidades salen de sus marcas. Si ya se
    limpió (en `_terminar`), se pasan `parrafos` y el `mapa` {id: [índices]}
    que dejó `_marcas_y_cobertura`, con la misma numeración."""
    if mapa is not None and parrafos is not None:
        inv: dict = {}
        for k, idxs in (mapa or {}).items():
            for i in (idxs or []):
                inv.setdefault(i, []).append(k)
        pars = [(p, inv.get(i, [])) for i, p in enumerate(parrafos)]
    else:
        pars = parrafos_con_unidades(estudio)
    fuera = []

    def _uno(fuente: str, clase: str, cita, extra: dict):
        usados = [(i, ids) for i, (txt, ids) in enumerate(pars) if cita(txt)]
        unidades = list(dict.fromkeys(x for _, ids in usados for x in ids))
        sust = any(_prefijo(x) in SUSTANTIVAS for x in unidades)
        fuera.append({"fuente": fuente, "clase": clase, "unidades": unidades,
                      "sustantiva": sust, "parrafos": [i for i, _ in usados],
                      "estado": ("justificacion_pendiente" if sust
                                 else ("sin_unidad" if usados else "no_usada")),
                      **extra})

    for t in (tesis_nuevas or []):
        if not isinstance(t, dict):
            continue
        try:
            import fuerza_juridica as _fj
            vinc = _fj.vincula(t)
        except Exception:                               # pragma: no cover
            vinc = None
        _uno(f"registro {t.get('registro')}", "tesis", lambda p, t=t: _cita_tesis(p, t),
             {"vincula": vinc, "citada_por_la_parte": bool(t.get("citada_por_la_parte"))})
    for n in (normas_nuevas or []):
        if not isinstance(n, dict):
            continue
        _uno(f"art. {n.get('articulo')} — {n.get('cuerpo_legal') or n.get('fuente') or '?'}",
             "norma", lambda p, n=n: _cita_norma(p, n),
             {"de_internet": bool(n.get("de_internet"))})
    return fuera


def informe_y_aviso(clasificadas: list) -> tuple:
    """(aviso o «», estado de salida o «»). El estado es «justificacion_pendiente»
    si alguna fuente tardía toca una unidad sustantiva."""
    sust = [c for c in (clasificadas or []) if c.get("sustantiva")]
    if not sust:
        return "", ""
    partes = []
    for c in sust[:6]:
        u = ", ".join(c["unidades"][:4])
        rasgo = (" (vincula a este tribunal)" if c.get("vincula") is True else "")
        rasgo += (" (texto de internet, sin verificar)" if c.get("de_internet") else "")
        partes.append(f"{c['fuente']}{rasgo} → {u}")
    aviso = ("JUSTIFICACIÓN PENDIENTE: estas fuentes se incorporaron DESPUÉS de redactar y "
             "se usan en razonamientos que deciden; esas unidades se escribieron sin tener su "
             "texto a la vista. Revisa cada una contra la fuente antes de firmar: "
             + "; ".join(partes) + ("…" if len(sust) > 6 else "") + ".")
    return aviso, "justificacion_pendiente"
