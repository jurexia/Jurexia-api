# -*- coding: utf-8 -*-
"""LA JUSTIFICACIÓN ESTRUCTURADA DE UNA SOLUCIÓN (rediseño del taller, etapa 3,
puntos 4 y 6 del plan de David, 29-sep-2026).

POR QUÉ. Hoy la propuesta trae, por solución, un párrafo: «el amparo debe
concederse porque…». Medido en el banco Kingston (diagnóstico del 29-sep): en
6 de 9 errores el motor tenía la información delante y aun así eligió
«fundado»; el párrafo no deja ver QUÉ razón de la responsable tenía que
superar, CON QUÉ regla, y QUÉ hecho acreditado la activa. Esta estructura lo
obliga a decirlo, y el código lo comprueba.

SolucionCandidata (un dict; se guarda así, nunca como dataclass):
  id, sentido, tipo_efecto, prospera                      ← de `soluciones.posibles`
  conclusion {texto, alcance}
  razones_a_superar [{razon (R del análisis neutral), caracter, como_se_supera}]
  regla {enunciado, fuentes [ids T/N/O del catálogo cerrado], fuerza, requisitos, excepciones}
  aplicacion [{requisito, hechos [{afirma, cita, fuente, condicion, verificada, lectura}]}]
  dificultad {objecion, respuesta}
  pendientes [{que, bloquea}]          ← «bloquea» sólo INFORMA (regla de David)
  efectos [{acto, parte, efecto}]      ← sólo si prospera (art. 77)
  precedente_propio [{id (O del catálogo), postura, requisito_o_hecho}]
  reparaciones []
  resumen (≤ 60 palabras)
  revision {avisos [...], faltan [...], estado}  ← lo calcula `revisar`, nunca el modelo

NADA BLOQUEA HASTA CALIBRAR (decisión 2): la revisión AVISA y marca estados
(«completa», «con_pendientes», «incompleta»); no cambia el sentido, no borra la
solución ni impide que el secretario la elija. Y lo que el diagnóstico midió:
exigir por regla dura que la solución que prospera supere TODAS las razones
autónomas tumba concesiones correctas (704, 296, 810: «un "además" no vuelve
autónoma una razón»), así que eso es AVISO, no filtro.

Puro: no llama al modelo. La cita se comprueba con la misma regla del
análisis neutral (`analisis_litis.verificar_cita`, calibrada contra OCR).
"""
from __future__ import annotations

import re

VERSION = "justificacion-1"
POSTURAS = ("sigue", "distingue", "se_aparta")
CARACTERES = ("autonoma", "conjunta", "dependiente")
MAX_RESUMEN = 60
MAX_LISTA = 8


def _txt(x, n: int = 0) -> str:
    s = " ".join(str(x or "").split())
    return s[:n] if n else s


def _lista(x) -> list:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _palabras(t: str, n: int) -> str:
    w = _txt(t).split(" ")
    return " ".join(w[:n]) if len(w) > n else _txt(t)


def normalizar(crudo, solucion: dict, *, cat: dict | None = None, T: dict | None = None) -> dict:
    """La SolucionCandidata limpia a partir de lo que devolvió el modelo para
    `solucion` (la enumerada por `soluciones.posibles`). Sólo forma y
    verificación de citas y fuentes; nunca lanza."""
    import analisis_litis as _al
    d = crudo if isinstance(crudo, dict) else {}
    s = solucion if isinstance(solucion, dict) else {}
    cat = cat or {}
    T = T or {}
    quitadas: list = []

    def _ids(x) -> list:
        out = []
        for k in re.findall(r"\b[TNO]\d{1,2}\b", " ".join(str(y) for y in _lista(x))):
            if k in cat and k not in out:
                out.append(k)
            elif k not in cat:
                quitadas.append(k)
        return out

    concl = d.get("conclusion") if isinstance(d.get("conclusion"), dict) else {"texto": d.get("conclusion")}
    regla = d.get("regla") if isinstance(d.get("regla"), dict) else {}
    dif = d.get("dificultad") if isinstance(d.get("dificultad"), dict) else {}

    aplicacion = []
    for a in _lista(d.get("aplicacion"))[:MAX_LISTA]:
        if not isinstance(a, dict):
            continue
        hechos = []
        for h in _lista(a.get("hechos"))[:MAX_LISTA]:
            if not isinstance(h, dict):
                continue
            cita, ok, fuente, lectura = _al.verificar_cita(h.get("cita"), T, h.get("fuente") or "acto") \
                if T else (_txt(h.get("cita"), 600), False, _al._fuente(h.get("fuente")), "")
            cond = _txt(h.get("condicion")).lower()
            cond = cond if cond in _al.CONDICIONES else "sin_verificar"
            # La misma regla que el análisis neutral: «falta en insumos» no
            # lleva cita; cualquier otra condición sin cita verificada no se da
            # por buena.
            if cond == "falta_en_insumos":
                if cita and ok:
                    cond = "sin_verificar"
                else:
                    cita, ok, lectura = "", True, ""
            elif not ok:
                cond = "sin_verificar"
            hechos.append({"afirma": _txt(h.get("afirma"), 500), "cita": cita, "fuente": fuente,
                           "condicion": cond, "verificada": bool(ok), "lectura": lectura})
        aplicacion.append({"requisito": _txt(a.get("requisito"), 400), "hechos": hechos})

    razones = []
    for r in _lista(d.get("razones_a_superar"))[:MAX_LISTA * 2]:
        if not isinstance(r, dict):
            continue
        car = _al._relacion(r.get("caracter")) if r.get("caracter") else "conjunta"
        razones.append({"razon": _txt(r.get("razon")).upper()[:8], "caracter": car,
                        "como_se_supera": _txt(r.get("como_se_supera"), 600)})

    precedentes = []
    for p in _lista(d.get("precedente_propio"))[:MAX_LISTA]:
        if not isinstance(p, dict):
            continue
        ids = _ids(p.get("id"))
        if not ids or cat.get(ids[0], {}).get("clase") != "propio":
            continue
        post = _txt(p.get("postura")).lower().replace(" ", "_")
        precedentes.append({"id": ids[0], "neun": cat[ids[0]].get("neun"),
                            "postura": post if post in POSTURAS else "distingue",
                            "requisito_o_hecho": _txt(p.get("requisito_o_hecho"), 400)})

    efectos = []
    if s.get("prospera"):
        for e in _lista(d.get("efectos"))[:MAX_LISTA]:
            if isinstance(e, dict) and _txt(e.get("efecto")):
                efectos.append({"acto": _txt(e.get("acto"), 300), "parte": _txt(e.get("parte"), 200),
                                "efecto": _txt(e.get("efecto"), 600)})

    out = {
        "version": VERSION,
        "id": s.get("id", ""), "sentido": s.get("sentido_rep", ""), "tipo_efecto": s.get("tipo_efecto", ""),
        "prospera": bool(s.get("prospera")),
        "conclusion": {"texto": _txt(concl.get("texto"), 900), "alcance": _txt(concl.get("alcance"), 400)},
        "razones_a_superar": razones,
        "regla": {"enunciado": _txt(regla.get("enunciado"), 900), "fuentes": _ids(regla.get("fuentes")),
                  "fuerza": _txt(regla.get("fuerza"), 60),
                  "requisitos": [_txt(x, 300) for x in _lista(regla.get("requisitos"))[:MAX_LISTA] if _txt(x)],
                  "excepciones": [_txt(x, 300) for x in _lista(regla.get("excepciones"))[:MAX_LISTA] if _txt(x)]},
        "aplicacion": aplicacion,
        "dificultad": {"objecion": _txt(dif.get("objecion"), 700), "respuesta": _txt(dif.get("respuesta"), 700)},
        "pendientes": [{"que": _txt(p.get("que"), 300), "bloquea": bool(p.get("bloquea"))}
                       for p in _lista(d.get("pendientes"))[:MAX_LISTA] if isinstance(p, dict) and _txt(p.get("que"))],
        "efectos": efectos,
        "precedente_propio": precedentes,
        "reparaciones": [_txt(x, 300) for x in _lista(d.get("reparaciones"))[:MAX_LISTA] if _txt(x)],
        "resumen": _palabras(d.get("resumen"), MAX_RESUMEN),
        "ids_quitados": len(quitadas),
    }
    return out


def revisar(sol: dict, *, analisis: dict | None = None, cat: dict | None = None) -> dict:
    """La revisión POR CÓDIGO de una SolucionCandidata normalizada: qué le
    falta para estar completa. Sólo AVISA (nada bloquea hasta calibrar) y
    devuelve {avisos, faltan, estado}. Nunca lanza."""
    avisos, faltan = [], []
    try:
        cat = cat or {}
        if not sol.get("conclusion", {}).get("texto"):
            faltan.append("conclusión")
        regla = sol.get("regla") or {}
        if not regla.get("enunciado"):
            faltan.append("regla")
        if not regla.get("fuentes"):
            avisos.append("La regla no cita ninguna fuente del catálogo: es una afirmación sin apoyo.")
        # CADA REQUISITO CON UN HECHO QUE LO ACTIVE, Y SU CITA VERIFICADA.
        reqs = regla.get("requisitos") or []
        aplicados = {_txt(a.get("requisito")).lower() for a in sol.get("aplicacion") or []}
        sin_aplicar = [r for r in reqs if _txt(r).lower() not in aplicados]
        if sin_aplicar:
            avisos.append(f"{len(sin_aplicar)} requisito(s) de la regla sin aplicación a los hechos: "
                          + "; ".join(sin_aplicar[:3]))
        for a in sol.get("aplicacion") or []:
            if not a.get("hechos"):
                avisos.append(f"El requisito «{a.get('requisito', '')[:80]}» no tiene ningún hecho que lo active.")
            for h in a.get("hechos") or []:
                if h.get("condicion") in ("sin_verificar", "no_acreditado", "falta_en_insumos"):
                    avisos.append(f"Hecho {h.get('condicion').replace('_', ' ')}: «{h.get('afirma', '')[:90]}».")
        # LA FUERZA INVOCADA TIENE QUE CASAR CON LA DEL CATÁLOGO (fuerza_juridica).
        f = (regla.get("fuerza") or "").lower()
        if f and regla.get("fuentes"):
            fuerzas = {str(cat.get(k, {}).get("fuerza") or "").lower() for k in regla["fuentes"]}
            if "obliga" in f and not any("oblig" in x for x in fuerzas):
                avisos.append("La regla se presenta como obligatoria y ninguna de sus fuentes obliga "
                              "a este tribunal (arts. 217 y 228 de la Ley de Amparo).")
        # LAS RAZONES DE LA RESPONSABLE: la solución que prospera tiene que
        # decir cómo supera las autónomas. AVISO, no filtro (ver cabecera).
        if sol.get("prospera") and isinstance(analisis, dict):
            autonomas = [r["id"] for r in analisis.get("razones") or []
                         if isinstance(r, dict) and r.get("relacion") == "autonoma"]
            tratadas = {r.get("razon") for r in sol.get("razones_a_superar") or [] if r.get("como_se_supera")}
            sin = [x for x in autonomas if x not in tratadas]
            if sin:
                avisos.append(f"La solución prospera sin decir cómo supera la(s) razón(es) autónoma(s) "
                              f"{', '.join(sin)} de la responsable: si alguna basta sola, derrotar las "
                              f"demás no cambia el resultado.")
        if sol.get("prospera") and sol.get("tipo_efecto") not in ("", "niega") and not sol.get("efectos"):
            avisos.append("La solución concede y no dice sus efectos (art. 77 de la Ley de Amparo).")
        if not sol.get("prospera") and sol.get("efectos"):
            avisos.append("La solución no prospera y trae efectos de concesión: se ignoran.")
        if not sol.get("dificultad", {}).get("objecion"):
            avisos.append("No enfrenta la objeción más fuerte en su contra.")
        if sol.get("ids_quitados"):
            avisos.append(f"Se quitaron {sol['ids_quitados']} fuente(s) que no están en el catálogo.")
        estado = "incompleta" if faltan else ("con_pendientes" if (avisos or sol.get("pendientes")) else "completa")
        return {"avisos": avisos, "faltan": faltan, "estado": estado}
    except Exception as ex:
        return {"avisos": [f"No se pudo revisar la solución ({type(ex).__name__})."], "faltan": [],
                "estado": "sin_revisar"}
