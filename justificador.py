# -*- coding: utf-8 -*-
"""EL JUSTIFICADOR POR SOLUCIÓN (rediseño del taller, etapa 3, punto 3; bandera
«soluciones_por_desenlace»).

QUÉ CAMBIA EN LA DELIBERACIÓN. Hoy argumentan dos abogados ciegos, uno por vía
(A prospera, B no). Con la bandera, argumenta UNO POR SOLUCIÓN enumerada por
código (`soluciones.posibles`: niega, reposición, para efectos, lisa y llana…),
cada uno ciego a los demás, en paralelo y con el mismo esfuerzo. El prompt es
el del abogado de siempre (`deliberacion.prompt_abogado`, probado) MÁS un
anexo propio de su solución: su desenlace y sus puntos resolutivos, sus
efectos, las razones de la responsable que tiene que superar (análisis
neutral, etapa 2) y los requisitos de la regla (demostración por requisitos,
etapa 2). Su respuesta se convierte en una SolucionCandidata
(`justificacion.normalizar`) y se revisa por código (`justificacion.revisar`).

LA TARJETA NO CAMBIA DE CONTRATO: sigue recibiendo dos vías, A y B. A es la
mejor solución que PROSPERA y B la mejor que NO (`elegir_vias`); todas van
además en `soluciones[]`, campo nuevo. «Mejor» es la de revisión más completa
y, a igualdad, la primera en orden del código: NUNCA la tasa base ni un
precedente (decisión 6 de David).

Aquí sólo lo puro; la llamada vive en `deliberacion.deliberar`.
"""
from __future__ import annotations

VERSION = "justificador-1"
_ORDEN_ESTADO = {"completa": 0, "con_pendientes": 1, "incompleta": 2, "sin_revisar": 3}


def _txt(x, n: int = 0) -> str:
    s = " ".join(str(x or "").split())
    return s[:n] if n else s


def anexo_solucion(sol: dict, analisis: dict | None = None, demostracion: dict | None = None) -> str:
    """El bloque que distingue a esta solución de las demás, con su esquema
    JSON adicional. «» si `sol` no sirve (se argumenta como hoy)."""
    if not isinstance(sol, dict) or not sol.get("id"):
        return ""
    L = ["", "═══ TU SOLUCIÓN (una de varias posibles; cada una la argumenta otra persona, ciega) ═══",
         f"SOLUCIÓN {sol['id']}: {'PROSPERA' if sol.get('prospera') else 'NO PROSPERA'} · "
         f"{str(sol.get('tipo_efecto') or '').replace('_', ' ')}"]
    if sol.get("nota_efecto"):
        L.append(f"  Qué significa: {sol['nota_efecto']}")
    if sol.get("desenlace"):
        L.append("  Puntos resolutivos que calculó el taller: " + " / ".join(sol["desenlace"][:3]))
    if sol.get("desenlace_nota"):
        L.append(f"  Nota: {sol['desenlace_nota']}")
    if sol.get("origen"):
        L.append(f"  Por qué cabe: {sol['origen']}")
    razones = [r for r in ((analisis or {}).get("razones") or []) if isinstance(r, dict)]
    if razones:
        L.append("  LAS RAZONES DE LO RESUELTO (análisis neutral; no califica):")
        for r in razones[:14]:
            L.append(f"   {r.get('id')} [{str(r.get('relacion') or '').upper()}] {_txt(r.get('afirma'), 300)}")
        if (analisis or {}).get("autonomas_sin_combatir"):
            L.append("   Autónomas que ningún argumento combate: " + ", ".join(analisis["autonomas_sin_combatir"]))
    reqs = [q for q in ((demostracion or {}).get("requisitos") or []) if isinstance(q, dict)]
    if reqs:
        L.append("  LOS REQUISITOS DE LA REGLA (demostración por requisitos):")
        for q in reqs[:6]:
            L.append(f"   {q.get('id')}. {_txt(q.get('enunciado'), 300)}")
    L.append("""  ADEMÁS de lo que se te pide arriba, devuelve en el MISMO JSON estas claves:
  "razones_a_superar": [{"razon": "<R… del análisis>", "caracter": "autonoma|conjunta|dependiente",
      "como_se_supera": "<cómo tu solución la vence o, si no prospera, por qué la sostiene>"}],
  "aplicacion": [{"requisito": "<cada requisito de la regla>", "hechos": [{"afirma": "<…>",
      "cita": "<literal, de cinco a cuarenta palabras>", "fuente": "acto|escrito|constancia",
      "condicion": "no_controvertido|tenido_por_acreditado|acreditacion_impugnada|no_acreditado|falta_en_insumos"}]}],
  "efectos": [{"acto": "<…>", "parte": "<…>", "efecto": "<…>"}]   (sólo si tu solución prospera),
  "pendientes": [{"que": "<lo que falta para cerrar esta solución>", "bloquea": false}],
  "conclusion_alcance": "<hasta dónde llega tu conclusión>",
  "resumen": "<sesenta palabras como máximo>"
  Si alguna razón AUTÓNOMA de la responsable no queda vencida, tu solución no puede prosperar
  por completo: dilo en `razones_a_superar` y en `sostenible`.""")
    return "\n".join(L) + "\n"


def a_candidata(crudo: dict, via: dict, sol: dict, *, cat: dict, T: dict, analisis: dict | None = None) -> dict:
    """La respuesta del justificador (ya verificada como vía por
    `deliberacion.verificar_via`) como SolucionCandidata revisada."""
    import justificacion as _ju
    c = crudo if isinstance(crudo, dict) else {}
    v = via if isinstance(via, dict) else {}
    cadena = v.get("cadena") or {}
    datos = {
        "conclusion": {"texto": v.get("razon") or cadena.get("conclusion") or c.get("conclusion"),
                       "alcance": c.get("conclusion_alcance")},
        "razones_a_superar": c.get("razones_a_superar"),
        "regla": {"enunciado": cadena.get("regla") or c.get("regla"),
                  "fuentes": [a.get("id") for a in (v.get("apoyos") or []) if isinstance(a, dict)]
                  or c.get("propongo_aplicar"),
                  "fuerza": next((a.get("fuerza") for a in (v.get("apoyos") or [])
                                  if isinstance(a, dict) and a.get("fuerza")), ""),
                  "requisitos": [a.get("requisito") for a in (c.get("aplicacion") or []) if isinstance(a, dict)],
                  "excepciones": []},
        "aplicacion": c.get("aplicacion") or ([{"requisito": "hechos de la vía", "hechos": cadena.get("hechos")}]
                                              if cadena.get("hechos") else []),
        "dificultad": {"objecion": (v.get("objecion") or {}).get("de_la_otra_via"),
                       "respuesta": (v.get("objecion") or {}).get("respuesta")},
        "pendientes": c.get("pendientes"),
        "efectos": c.get("efectos"),
        "precedente_propio": [{"id": p.get("id"), "postura": p.get("trato"), "requisito_o_hecho": p.get("por_que")}
                              for p in (c.get("precedente_propio") or []) if isinstance(p, dict)],
        "resumen": c.get("resumen") or v.get("razon"),
    }
    s = _ju.normalizar(datos, sol, cat=cat, T=T)
    s["sostenible"] = bool(c.get("sostenible", True))
    s["respondio"] = bool(v.get("respondio"))
    s["rama"], s["desenlace"] = sol.get("rama", ""), list(sol.get("desenlace") or [])
    s["revision"] = _ju.revisar(s, analisis=analisis, cat=cat)
    if not s["sostenible"]:
        s["revision"]["avisos"].insert(0, "Su propio justificador dice que no se puede sostener con estas fuentes y hechos.")
    return s


def elegir_vias(candidatas: list) -> tuple:
    """(id de la vía A, id de la vía B): la mejor que prospera y la mejor que
    no, por la revisión y luego por el orden del código. Nunca por tasa base."""
    def mejor(lado: bool):
        cs = [c for c in candidatas or [] if isinstance(c, dict) and bool(c.get("prospera")) == lado
              and c.get("respondio")]
        if not cs:
            return None
        cs.sort(key=lambda c: (not c.get("sostenible", True),
                               _ORDEN_ESTADO.get((c.get("revision") or {}).get("estado"), 3),
                               int(str(c.get("id", "S9"))[1:] or 9)))
        return cs[0]["id"]
    return mejor(True), mejor(False)
