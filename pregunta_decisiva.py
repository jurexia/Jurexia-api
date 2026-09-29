# -*- coding: utf-8 -*-
"""LA PREGUNTA DECISIVA, PARA TODOS — SPEC E3, 28-sep-2026.

POR QUÉ EXISTE. David, tras leer el proyecto intermedio del AR 631/2025: «aunque
la cosa juzgada ciertamente es un tema que se aborda en la sentencia recurrida,
lo jurídicamente relevante es la posibilidad de que un tercero se subrogue o
sustituya en los derechos de ejecución de esa cosa juzgada. El debate no va en
torno a si se infringe la cosa juzgada, sino a la posibilidad jurídica de que un
tercero adquirente del inmueble objeto del juicio natural (que versó sobre una
acción personal) pueda válidamente sustituirse en ejecución (para eso el RAG y
la búsqueda en internet de fuentes y, en su caso, de la interpretación
conforme)».

El motor tomó el problema tal como lo formuló la recurrida y con eso buscó y
razonó: la búsqueda trajo dos jurisprudencias genéricas de cosa juzgada, la
propuesta razonó sobre la cosa juzgada y el estudio también. La formulación
manda (medido en `fase6_rag`: consulta conceptual contra el vector `rubro`, 50 %
en primera posición; prosa, 3 %).

QUÉ HACE. La ETAPA A de `deliberacion.py` —ya integrada, detrás de su bandera—
corre aquí SIEMPRE, para todos, y barata (modelo de lectura, esfuerzo bajo, una
llamada): devuelve la figura, la pregunta decisiva, la proposición toral con su
cita literal VERIFICADA contra el acto, los hechos que deciden, las consultas
sobre la figura en lenguaje de rubro y, si la respuesta depende del alcance de
un precepto, la interpretación conforme que hay que examinar. Se guarda como
marca «decisiva» con la huella del adelanto (gunicorn -w 2: nada en memoria) y
viaja con el material (`material.decisiva`), que es lo que leen todos:

  · RAG: las consultas sobre la figura SE SUMAN al material con cupo propio y
    marcadas `para` el principal (`fase6_rag.tesis_de_la_figura` y
    `sumar_figura`); lo de hoy no se toca.
  · internet: `fase_internet.precedentes_verificados` se apunta a la pregunta
    decisiva y, si la trae, a la interpretación conforme; sólo entra lo que el
    acervo confirma; las pistas no se citan (`pregunta_internet`).
  · propuesta: el principal se presenta con la cuestión decisiva y la figura,
    y la pregunta de la recurrida va como dato (`bloque_propuesta`). NO se
    sube el `formato` de la propuesta: recalcularía todas las guardadas.
  · estudio: el criterio del principal y el guion llevan «LA CUESTIÓN
    DECISIVA» como dato (`lineas_criterio`, `lineas_guion`).
  · tarjeta: `principal.pregunta` es la decisiva y `pregunta_recurrida` va
    aparte (`para_tarjeta`).
  · deliberación: si está encendida, reutiliza esta misma pregunta en vez de
    volver a pagarla (`deliberacion.deliberar(decisiva_previa=…)`).

CUÁNDO CORRE, Y POR QUÉ AHÍ. En `_taller_preconsultar`, EN PARALELO con la
consulta del acervo, en cuanto termina el adelanto; y la búsqueda de la figura,
al terminar la consulta, antes del refuerzo y de la propuesta. La otra opción
era dentro de la propuesta, y se descartó: la propuesta se lanza al terminar la
consulta, así que la figura tendría que buscarse ahí, en serie, y el material
guardado (el que lee el estudio en el otro worker) no la tendría. En paralelo,
la espera añadida es sólo lo que la pregunta tarde MÁS que la consulta (que
lleva una traducción conceptual y un rerank por problema, decenas de segundos)
más la búsqueda de la figura (embeddings y Qdrant, sin modelo). Las dos cosas
se miden en el código y se imprimen («🎯 PREGUNTA DECISIVA …»).

COSTE. Una llamada corta al modelo de lectura (≈ 0.002-0.01 USD por asunto,
contado en `uso`) más de dos a cuatro embeddings y consultas a Qdrant. Las
pruebas lo cuentan con modelos falsos (test_pregunta_decisiva.py), sin red.

LO QUE NO HACE: no decide (no dice quién gana ni califica); no inventa registros
(la búsqueda devuelve lo que hay en el acervo); no sustituye la pregunta que
viaja en la fase 3 (la del secretario sigue mandando en el emparejamiento y en
la jerarquía); y si la pregunta no se formuló o ya no es la del principal
—el secretario cambió el principal o su pregunta—, nadie la usa.

BANDERA. `PREGUNTA_DECISIVA_ACTIVA`: encendida por omisión; «0» la apaga y
entonces este módulo no hace ninguna llamada y todo queda como antes.
"""
from __future__ import annotations

import os
import time
import unicodedata
from typing import Any, Optional

FORMATO = 1
VERSION = "decisiva-1"
CLAVE_MARCA = "decisiva"


def activa() -> bool:
    """Se lee en cada llamada: se apaga sin reiniciar el proceso."""
    return (os.getenv("PREGUNTA_DECISIVA_ACTIVA", "1") or "1").strip().lower() not in (
        "0", "off", "no", "false", "apagada")


def _txt(x: Any) -> str:
    return " ".join(str(x or "").split())


def _plano(x: Any) -> str:
    t = unicodedata.normalize("NFD", _txt(x).lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return " ".join("".join(c if c.isalnum() else " " for c in t).split())


# ═══ LA LLAMADA (la ETAPA A de la deliberación, reutilizada) ═════════════════

def entradas_de(r) -> dict:
    """Lo que la pregunta lee del adelanto: los planteamientos con la misma
    construcción que la huella (`taller_estado.entradas_contraste`), los dos
    resúmenes y el texto literal de la resolución (de ahí sale la cita)."""
    import taller_estado as _te
    problemas, acto, conceptos, es_recurso = _te.entradas_contraste(r)
    f = getattr(r, "fases", None)
    e = getattr(r, "encargo", None)
    fuentes = list(getattr(f, "fuentes", None) or []) + [""]
    return {"problemas": problemas, "resumen_acto": acto, "resumen_conceptos": conceptos,
            "texto_acto": str(fuentes[0] or ""), "es_recurso": bool(es_recurso),
            "tipo_asunto": str(getattr(e, "tipo_asunto", "") or "") if e is not None else ""}


async def formular(cliente, *, problemas: list, resumen_acto: str = "",
                   resumen_conceptos: str = "", texto_acto: str = "",
                   tipo_asunto: str = "", es_recurso: bool = False,
                   contraste: Optional[dict] = None) -> dict:
    """La pregunta decisiva del principal, como documento de la marca. Nunca
    lanza por el modelo: si la llamada falla, `formulada` es False y nadie la
    usa (se sigue con la pregunta de la fase 3, como antes)."""
    import deliberacion as _dl
    t0 = time.perf_counter()
    uso, avisos = _dl._Uso(), []
    probs = [dict(p) if isinstance(p, dict) else {"pregunta": str(p)} for p in (problemas or [])]
    base = {"formato": FORMATO, "version": VERSION, "origen": "pregunta_decisiva"}
    if not probs:
        return dict(base, formulada=False, avisos=["No hay planteamientos."], uso=uso.doc(),
                    segundos=0.0)
    pi = _dl.indice_principal(probs)
    tx = _dl._Textos({"acto": texto_acto})
    d = await _dl.etapa_a(cliente, probs[pi], contraste, resumen_acto, resumen_conceptos, tx,
                          tipo_asunto, es_recurso, uso, avisos)
    doc = dict(base, numero=pi + 1, tipo_asunto=tipo_asunto, es_recurso=bool(es_recurso), **d)
    doc.update(avisos=avisos, uso=uso.doc(), segundos=round(time.perf_counter() - t0, 1))
    return doc


def marca(doc: dict, huella: str, estado: str = "listo") -> dict:
    """La marca «decisiva» en la fila: {huella, estado, doc}."""
    return {"huella": huella or "", "estado": estado, "doc": doc}


def doc_de_marca(m: Any, huella: str = "") -> Optional[dict]:
    """El documento de la marca si está lista y es de ESTE adelanto."""
    if not isinstance(m, dict) or m.get("estado") != "listo":
        return None
    if huella and m.get("huella") != huella:
        return None
    d = m.get("doc")
    return d if util(d) else None


# ═══ QUIÉN LA USA, Y CUÁNDO VALE ═════════════════════════════════════════════

def util(doc: Any) -> bool:
    return (isinstance(doc, dict) and bool(doc.get("formulada"))
            and bool(_txt(doc.get("pregunta_decisiva"))))


def _principal(problemas: list) -> tuple:
    import deliberacion as _dl
    probs = [p if isinstance(p, dict) else {"pregunta": str(p)} for p in (problemas or [])]
    if not probs:
        return -1, {}
    i = _dl.indice_principal(probs)
    return i, probs[i]


def vigente(doc: Any, problemas: Optional[list] = None) -> Optional[dict]:
    """El documento si sirve para el principal de HOY. Se formuló sobre la
    pregunta del principal tal como estaba; si el secretario cambió de
    principal o corrigió su pregunta, ya no es la cuestión de ese problema y
    no se usa (mejor ninguna que la de otro problema). Sin `problemas`, basta
    con que esté formulada."""
    if not util(doc):
        return None
    if problemas is None:
        return doc
    _, pral = _principal(problemas)
    if not pral:
        return None
    rec = _plano(doc.get("pregunta_recurrida"))
    candidatas = {_plano(pral.get("pregunta")), _plano(pral.get("pregunta_original"))}
    return doc if rec and rec in candidatas else None


def de_material(material: Any, problemas: Optional[list] = None) -> Optional[dict]:
    d = (material.get("decisiva") if isinstance(material, dict)
         else getattr(material, "decisiva", None))
    return vigente(d, problemas)


def consultas_rag(doc: Any) -> list:
    """Las consultas sobre la figura para el vector `rubro`. Si el modelo no
    dio ninguna, la figura sola (es un sintagma nominal: sirve contra el
    rubro). La pregunta decisiva NO va: es prosa interrogativa."""
    if not util(doc):
        return []
    qs = [q for q in (doc.get("busquedas") or []) if _txt(q)]
    if not qs and _txt(doc.get("figura")):
        qs = [_txt(doc.get("figura"))]
    return qs[:4]


def numero(doc: Any) -> int:
    try:
        return int((doc or {}).get("numero") or 0)
    except (TypeError, ValueError):
        return 0


def pregunta_internet(doc: Any, respaldo: str = "") -> str:
    """La pregunta con que se busca la línea de la Corte en internet: la
    decisiva, su figura y, si la trae, el precepto cuya interpretación conforme
    hay que examinar. Sin pregunta decisiva, la de siempre."""
    if not util(doc):
        return respaldo
    partes = [_txt(doc.get("pregunta_decisiva"))]
    if _txt(doc.get("figura")):
        partes.append(f"Figura: {_txt(doc.get('figura'))}.")
    ic = doc.get("interpretacion_conforme") if isinstance(doc.get("interpretacion_conforme"), dict) else None
    if ic and _txt(ic.get("precepto")):
        partes.append(f"Interpretación conforme de {_txt(ic.get('precepto'))}"
                      + (f": {_txt(ic.get('por_que'))}" if _txt(ic.get("por_que")) else "") + ".")
    return " ".join(partes)


def _como_llego(doc: dict) -> str:
    return ("ASÍ LO PLANTEÓ LA RECURRIDA" if doc.get("es_recurso")
            else "ASÍ LLEGÓ PLANTEADO")


def bloque_propuesta(doc: Any) -> str:
    """Bloque de DATOS para el prompt de la propuesta. Describe qué es cada
    renglón y cómo se usa; no trae ninguna frase para copiar."""
    if not util(doc):
        return ""
    n = numero(doc) or 1
    tor = doc.get("proposicion_toral") if isinstance(doc.get("proposicion_toral"), dict) else {}
    L = ["", f"LA CUESTIÓN DECISIVA DEL PROBLEMA {n} (el principal) — dato del asunto, no conclusión",
         f"· {_como_llego(doc)}: {_txt(doc.get('pregunta_recurrida')) or '(no consta)'}",
         f"· LA CUESTIÓN QUE DECIDE: {_txt(doc.get('pregunta_decisiva'))}"]
    if _txt(doc.get("figura")):
        L.append(f"· LA FIGURA: {_txt(doc.get('figura'))}")
    if _txt(tor.get("dice")):
        L.append(f"· PROPOSICIÓN TORAL DE LA RESOLUCIÓN: {_txt(tor.get('dice'))}"
                 + (f" — literal: «{_txt(tor.get('cita'))}»" if _txt(tor.get("cita")) else ""))
    hs = [h for h in (doc.get("hechos_que_deciden") or []) if _txt(h)]
    if hs:
        L.append("· HECHOS DE LOS QUE DEPENDE: " + " | ".join(_txt(h) for h in hs))
    ic = doc.get("interpretacion_conforme") if isinstance(doc.get("interpretacion_conforme"), dict) else None
    if ic and _txt(ic.get("precepto")):
        L.append(f"· INTERPRETACIÓN CONFORME POR EXAMINAR: {_txt(ic.get('precepto'))}"
                 + (f" — {_txt(ic.get('por_que'))}" if _txt(ic.get("por_que")) else ""))
    L += [f"Cómo se usa: el problema {n} se presenta y se resuelve contestando LA CUESTIÓN QUE",
          "DECIDE; la pregunta tal como llegó planteada es el marco en que se contesta, no",
          "la que se resuelve. La propuesta global razona sobre esa cuestión. Las tesis",
          "marcadas «de la figura» se buscaron para ella.", ""]
    return "\n".join(L)


def lineas_criterio(doc: Any) -> list:
    """Los renglones que el criterio del principal lleva en el estudio."""
    if not util(doc):
        return []
    L = [f"   LA CUESTIÓN DECISIVA: {_txt(doc.get('pregunta_decisiva'))}"]
    if _txt(doc.get("figura")):
        L.append(f"   FIGURA: {_txt(doc.get('figura'))}")
    if _txt(doc.get("pregunta_recurrida")):
        L.append(f"   {_como_llego(doc)}: {_txt(doc.get('pregunta_recurrida'))}")
    L.append("   EL ESTUDIO DE ESTE PROBLEMA SE ORGANIZA EN TORNO A LA CUESTIÓN DECISIVA: lo "
             "planteado así es el marco, y se contesta desde ella.")
    return L


def lineas_guion(doc: Any) -> list:
    """El renglón del guion (datos, sin prosa)."""
    if not util(doc):
        return []
    return [f"LA CUESTIÓN DECISIVA (problema {numero(doc) or 1}): {_txt(doc.get('pregunta_decisiva'))}"
            + (f" · figura: {_txt(doc.get('figura'))}" if _txt(doc.get("figura")) else "")
            + (f" · planteada así: {_txt(doc.get('pregunta_recurrida'))}"
               if _txt(doc.get("pregunta_recurrida")) else "")
            + " · el apartado de ese problema se organiza en torno a ella"]


def para_tarjeta(doc: Any, pregunta_fase3: str) -> dict:
    """{pregunta, pregunta_recurrida, figura} para `principal` de la tarjeta.
    Sin pregunta decisiva vigente, la pregunta de la fase 3 y nada aparte."""
    if not util(doc):
        return {"pregunta": pregunta_fase3, "pregunta_recurrida": None, "figura": None}
    return {"pregunta": _txt(doc.get("pregunta_decisiva")),
            "pregunta_recurrida": _txt(doc.get("pregunta_recurrida")) or pregunta_fase3,
            "figura": _txt(doc.get("figura")) or None}
