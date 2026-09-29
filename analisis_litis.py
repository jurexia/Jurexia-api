# -*- coding: utf-8 -*-
"""EL ANÁLISIS NEUTRAL DE LA LITIS, ANTES DE DECIDIR (rediseño del taller,
punto 1 y etapa 2; David, 29-sep-2026).

POR QUÉ. David: «Hoy plan_estudio mezcla dos trabajos: comprender qué debe
resolverse y organizar cómo escribir una decisión elegida. Separaría esos
productos.» Lo más parecido a un análisis previo era el contraste, que no es
neutral (precalifica), lee sólo los resúmenes y reduce la recurrida a UNA razón
toral; las proposiciones P de la recurrida las extraía el PLANIFICADOR, después
de decidir, y salían distintas con cada criterio.

QUÉ PRODUCE (una llamada con razonamiento + una verificación SIN modelo):
  · cuestion_central — la cuestión de derecho concreta de la que depende el
    asunto, sin reducirla a una pregunta general (en el AR 631/2025: la
    transmisión del derecho contractual concreto, no «¿se alteró la cosa
    juzgada?»);
  · razones R — lo que sostiene la recurrida (o el acto): qué afirma, qué
    concluye, y si funciona SOLA (autónoma: basta para sostener el fallo),
    JUNTO con otras (conjunta: premisas necesarias a la vez) o DEPENDIENTE de
    otra; con su cita LITERAL del acto;
  · argumentos — cada segmento del inventario (C1.a, A2.b…): qué pide, por
    qué, y qué razones R combate;
  · hechos H — qué ocurrió, quién lo afirma, dónde consta, con cita, y su
    CONDICIÓN PROBATORIA, que David pidió distinguir porque confundirlas cambia
    el sentido:
        falta_en_insumos        el documento no está en lo que se subió;
        no_acreditado           la parte no lo acreditó en el juicio;
        tenido_por_acreditado   el órgano recurrido lo tuvo por acreditado;
        acreditacion_impugnada  esa determinación probatoria está impugnada;
        no_controvertido        consta y nadie lo discute;
  · faltantes — los insumos que harían falta para decidir.
  · Lo REVISABLE y lo FIRME no se deducen aquí: vienen de la ficha procesal,
    que ya los calcula por código, y se copian.

POR CÓDIGO, NO POR EL MODELO:
  · toda cita se comprueba palabra por palabra contra su fuente
    (`plan_estudio.Texto`); la que no está deja de presentarse como cita
    (`verificada: False`) y un hecho «tenido por acreditado» sin cita
    verificada baja a «sin_verificar»;
  · las referencias (combate, con, segmento) que no existen se quitan;
  · `autonomas_sin_combatir`: las razones autónomas que ningún argumento
    combate. Es el punto 5 de David («si R1 y R2 sostienen autónomamente la
    sentencia, derrotar R1 no elimina R2»), medido sobre el análisis y no
    adivinado al redactar.

LO QUE NO ENTRA. Ni el acervo ni los precedentes de la OAJ: traen calificación y
razón de otros asuntos y empujarían el sentido antes de decidir. El análisis es
de ESTE expediente. Tampoco califica: no dice fundado ni infundado.

NUNCA BLOQUEA. Si falla o tarda, la propuesta se hace sin él (regla de David:
el secretario siempre puede pedir la propuesta al motor) y se dice.
"""
from __future__ import annotations

import hashlib
import json
import os
import re

VERSION = "analisis-1"
MAX_ACTO = 60000
MAX_ESCRITO = 60000
MAX_AUTOS = 20000
MAX_TOKENS = int(os.getenv("ANALISIS_MAX_TOKENS", "24000"))
MODELO = os.getenv("MODELO_ANALISIS", "") or None     # vacío = el de la propuesta
CLAVE_MARCA = "analisis"

CONDICIONES = ("falta_en_insumos", "no_acreditado", "tenido_por_acreditado",
               "acreditacion_impugnada", "no_controvertido")
RELACIONES = ("autonoma", "conjunta", "dependiente")
_RX_JSON = re.compile(r"\{.*\}", re.S)


def _txt(x, n: int = 0) -> str:
    s = " ".join(str(x or "").split())
    return s[:n] if n else s


def huella(r) -> str:
    """Lo que el análisis LEE: el acto, el escrito, las constancias, los
    planteamientos y la versión. Si cambia uno, el análisis ya no es de este
    adelanto."""
    f = getattr(r, "fases", None)
    fuentes = list(getattr(f, "fuentes", []) or []) + ["", ""]
    probs = [(_txt((p or {}).get("pregunta")), _txt((p or {}).get("combate")), _txt((p or {}).get("resolvio")))
             for p in (getattr(f, "problemas", None) or []) if isinstance(p, dict)]
    base = json.dumps([VERSION, str(fuentes[0])[:MAX_ACTO], str(fuentes[1])[:MAX_ESCRITO],
                       str(getattr(f, "autos", "") or "")[:MAX_AUTOS], probs], ensure_ascii=False)
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:20]


# ═══ EL PROMPT ═══════════════════════════════════════════════════════════════

def prompt(acto: str, escrito: str, autos: str, segmentos: list, problemas: list,
           ficha: str, es_recurso: bool, tipo_asunto: str = "") -> str:
    q = "agravio" if es_recurso else "concepto de violación"
    recurrida = "la sentencia recurrida" if es_recurso else "el acto reclamado"
    segs = "\n".join(f"  {s.get('id')}: {_txt(s.get('texto'), 600)}" for s in (segmentos or [])
                     if isinstance(s, dict) and s.get("id")) or "  (sin inventario)"
    probs = "\n".join(f"  {i}. {_txt((p or {}).get('pregunta'), 400)}"
                      for i, p in enumerate(problemas or [], 1) if isinstance(p, dict)) or "  (sin planteamientos)"
    return f"""Eres secretario de estudio y cuenta de un Tribunal Colegiado de Circuito.
ANTES de decidir nada, analiza la litis de este asunto ({tipo_asunto or 'sin tipo'}). NO
califiques ni propongas sentido: sólo describe con precisión lo que hay que resolver.

LO QUE DEBES ENTREGAR, en JSON, con estas claves:

cuestion_central: la cuestión de derecho CONCRETA de la que depende el asunto, en una
  oración. No la reduzcas a una pregunta general: nombra la figura, el derecho y el acto
  concretos que están en juego.

razones: lista de lo que sostiene {recurrida}. Cada una:
  id ("R1", "R2"…), afirma (qué dice), conclusion (qué concluye con ello),
  relacion: "autonoma" si basta SOLA para sostener lo resuelto; "conjunta" si sólo
  funciona junto con otras (di con cuáles en "con"); "dependiente" si presupone otra
  (di cuál en "con"),
  con: ids de las razones con que se relaciona,
  problemas: números de los planteamientos a que se refiere,
  cita: 10 a 40 palabras COPIADAS LITERALMENTE de {recurrida} donde lo dice.

argumentos: uno por cada segmento del inventario de abajo, con su id tal cual:
  segmento (el id), pide (qué pide), por_que (por qué lo pide),
  combate: ids de las razones R que ataca ([] si no ataca ninguna).

hechos: los hechos que importan para decidir. Cada uno:
  id ("H1"…), que (qué ocurrió), afirma (quién lo afirma: quejoso, recurrente,
  autoridad, a_quo, tercero o constancias), fuente ("acto", "escrito" o "constancias"),
  cita: 10 a 40 palabras LITERALES de esa fuente ("" si el hecho no consta en los insumos),
  condicion, UNA de estas, y distínguelas con cuidado porque confundirlas cambia el sentido:
    "falta_en_insumos": el documento o dato NO está en lo que se entregó (no confundir
       con que la parte no lo haya probado);
    "no_acreditado": la parte no lo acreditó en el juicio;
    "tenido_por_acreditado": el órgano que resolvió lo tuvo por acreditado;
    "acreditacion_impugnada": esa determinación probatoria se combate en el {q};
    "no_controvertido": consta y nadie lo discute.

faltantes: lista de {{"que": …, "por_que_importa": …}} con lo que haría falta tener para
  decidir y no está en los insumos. [] si no falta nada.

REGLAS:
- Copia las citas palabra por palabra; si no encuentras el pasaje, deja la cita en "".
- No inventes hechos ni razones que no estén en los textos.
- Lo revisable y lo firme ya está en la ficha procesal: no lo repitas ni lo contradigas.
- No califiques: nada de fundado, infundado, inoperante ni sentido propuesto.

FICHA PROCESAL (datos, calculados por código):
{ficha or '(sin ficha)'}

PLANTEAMIENTOS:
{probs}

INVENTARIO DE ARGUMENTOS DEL ESCRITO (usa estos ids):
{segs}

{recurrida.upper()}:
<<<
{str(acto or '')[:MAX_ACTO]}
>>>

ESCRITO DE LA PARTE ({q}s):
<<<
{str(escrito or '')[:MAX_ESCRITO]}
>>>

CONSTANCIAS:
<<<
{str(autos or '')[:MAX_AUTOS] or '(no se entregaron)'}
>>>

Responde SÓLO con el JSON."""


# ═══ LA VERIFICACIÓN, SIN MODELO ═════════════════════════════════════════════

def verificar(crudo: dict, acto: str, escrito: str, autos: str, segmentos: list) -> dict:
    """El análisis limpio: citas comprobadas, referencias válidas, condiciones
    del catálogo y las razones autónomas sin combatir calculadas por código."""
    import plan_estudio as _pe
    T = {"acto": _pe.Texto(acto or ""), "escrito": _pe.Texto(escrito or ""),
         "constancias": _pe.Texto(autos or "")}
    d = crudo if isinstance(crudo, dict) else {}

    def _cita(c: str, fuente: str) -> tuple:
        c = _txt(c, 600)
        if not c:
            return "", False
        t = T.get(fuente) or T["acto"]
        if t.contiene(c):
            return c, True
        rec, _ = _pe._recorte_literal(c, [t])
        return (rec, True) if rec else (c, False)

    razones, ids_r = [], set()
    for i, x in enumerate(d.get("razones") or [], 1):
        if not isinstance(x, dict):
            continue
        rid = _txt(x.get("id")) or f"R{i}"
        if rid in ids_r:
            continue
        ids_r.add(rid)
        rel = _txt(x.get("relacion")).lower()
        cita, ok = _cita(x.get("cita"), "acto")
        razones.append({"id": rid, "afirma": _txt(x.get("afirma"), 800),
                        "conclusion": _txt(x.get("conclusion"), 600),
                        "relacion": rel if rel in RELACIONES else "autonoma",
                        "con": [_txt(c) for c in (x.get("con") or []) if _txt(c)],
                        "problemas": [int(p) for p in (x.get("problemas") or [])
                                      if str(p).strip().isdigit()],
                        "cita": cita, "verificada": ok})
    for r_ in razones:
        r_["con"] = [c for c in r_["con"] if c in ids_r and c != r_["id"]]

    ids_seg = {str(s.get("id")) for s in (segmentos or []) if isinstance(s, dict) and s.get("id")}
    argumentos, vistos = [], set()
    for x in (d.get("argumentos") or []):
        if not isinstance(x, dict):
            continue
        sid = _txt(x.get("segmento"))
        if ids_seg and sid not in ids_seg:
            continue
        if sid in vistos:
            continue
        vistos.add(sid)
        argumentos.append({"segmento": sid, "pide": _txt(x.get("pide"), 500),
                           "por_que": _txt(x.get("por_que"), 600),
                           "combate": [c for c in (_txt(y) for y in (x.get("combate") or [])) if c in ids_r]})

    hechos = []
    for i, x in enumerate(d.get("hechos") or [], 1):
        if not isinstance(x, dict):
            continue
        fuente = _txt(x.get("fuente")).lower()
        fuente = fuente if fuente in T else "acto"
        cond = _txt(x.get("condicion")).lower()
        cond = cond if cond in CONDICIONES else "sin_verificar"
        cita, ok = _cita(x.get("cita"), fuente)
        # LA CONDICIÓN SE SOSTIENE CON SU FUENTE: «falta en insumos» no lleva cita;
        # cualquier otra condición sin cita verificada no se da por buena.
        if cond == "falta_en_insumos":
            cita, ok = "", True
        elif not ok:
            cond = "sin_verificar"
        hechos.append({"id": _txt(x.get("id")) or f"H{i}", "que": _txt(x.get("que"), 600),
                       "afirma": _txt(x.get("afirma"), 60), "fuente": fuente,
                       "cita": cita, "verificada": ok, "condicion": cond})

    faltantes = [{"que": _txt(x.get("que"), 300), "por_que_importa": _txt(x.get("por_que_importa"), 400)}
                 for x in (d.get("faltantes") or []) if isinstance(x, dict) and _txt(x.get("que"))]

    combatidas = {c for a in argumentos for c in a["combate"]}
    autonomas_sin = [r_["id"] for r_ in razones if r_["relacion"] == "autonoma" and r_["id"] not in combatidas]
    return {"version": VERSION, "cuestion_central": _txt(d.get("cuestion_central"), 600),
            "razones": razones, "argumentos": argumentos, "hechos": hechos, "faltantes": faltantes,
            "autonomas_sin_combatir": autonomas_sin,
            "citas_sin_verificar": sum(1 for x in razones + hechos if x.get("cita") and not x.get("verificada"))}


# ═══ LA LLAMADA ══════════════════════════════════════════════════════════════

async def analizar(cliente, r, segmentos: list, ficha: str = "") -> dict | None:
    """El análisis de este adelanto, o None si falla (nunca lanza)."""
    try:
        import fase5_propuesta as _f5
        import llamada_modelo as _lm
        f = r.fases
        fuentes = list(getattr(f, "fuentes", []) or []) + ["", ""]
        acto, escrito = str(fuentes[0] or ""), str(fuentes[1] or "")
        autos = str(getattr(f, "autos", "") or "")
        e = getattr(r, "encargo", None)
        es_rec = bool(e is not None and getattr(e, "es_recurso", False))
        tipo = str(getattr(e, "tipo_asunto", "") or "")
        probs = [p for p in (getattr(f, "problemas", None) or []) if isinstance(p, dict)]
        kw = dict(model=MODELO or _f5.MODELO_PROPUESTA, temperature=0, seed=20260929,
                  max_completion_tokens=MAX_TOKENS,
                  messages=[{"role": "user", "content": prompt(
                      acto, escrito, autos, segmentos, probs, ficha, es_rec, tipo)}])
        if _f5.ESFUERZO_PROPUESTA:
            kw["reasoning_effort"] = _f5.ESFUERZO_PROPUESTA
        resp = await _lm.crear(cliente, **kw)
        crudo = (resp.choices[0].message.content or "").strip()
        if not crudo:
            print("   🔎 ANÁLISIS: volvió vacío; se propone sin él")
            return None
        m = _RX_JSON.search(crudo)
        if not m:
            print("   🔎 ANÁLISIS: sin JSON; se propone sin él")
            return None
        doc = verificar(json.loads(m.group(0)), acto, escrito, autos, segmentos)
        print(f"   🔎 ANÁLISIS NEUTRAL: {len(doc['razones'])} razones "
              f"({len(doc['autonomas_sin_combatir'])} autónomas sin combatir) · "
              f"{len(doc['argumentos'])} argumentos · {len(doc['hechos'])} hechos · "
              f"{len(doc['faltantes'])} faltantes · {doc['citas_sin_verificar']} citas sin verificar")
        return doc
    except Exception as ex:
        print(f"   🔎 ANÁLISIS: falló ({type(ex).__name__}); se propone sin él")
        return None


# ═══ PARA LA PROPUESTA ═══════════════════════════════════════════════════════

_TXT_COND = {"falta_en_insumos": "NO CONSTA en los insumos (no es lo mismo que no probado)",
             "no_acreditado": "la parte NO lo acreditó",
             "tenido_por_acreditado": "el órgano recurrido lo TUVO POR ACREDITADO",
             "acreditacion_impugnada": "su acreditación ESTÁ IMPUGNADA",
             "no_controvertido": "consta y no se controvierte",
             "sin_verificar": "condición SIN VERIFICAR (su cita no se halló)"}


def bloque_propuesta(doc: dict | None) -> str:
    """El análisis como DATOS para la propuesta, con su regla. «» si no hay."""
    if not isinstance(doc, dict) or not (doc.get("razones") or doc.get("hechos")):
        return ""
    L = ["", "ANÁLISIS NEUTRAL DE LA LITIS (datos previos a decidir; no califica):"]
    if doc.get("cuestion_central"):
        L.append(f"  CUESTIÓN CENTRAL: {doc['cuestion_central']}")
    if doc.get("razones"):
        L.append("  RAZONES DE LO RESUELTO:")
        for r_ in doc["razones"]:
            rel = {"autonoma": "AUTÓNOMA (basta sola)", "conjunta": "CONJUNTA", "dependiente": "DEPENDIENTE"}[r_["relacion"]]
            con = f" con {', '.join(r_['con'])}" if r_.get("con") else ""
            L.append(f"   {r_['id']} [{rel}{con}] {r_['afirma']} → {r_['conclusion']}"
                     + (f"\n       cita{'' if r_['verificada'] else ' NO verificada'}: «{r_['cita']}»" if r_.get("cita") else ""))
    comb = {}
    for a in doc.get("argumentos") or []:
        for c in a["combate"]:
            comb.setdefault(c, []).append(a["segmento"])
    if doc.get("razones"):
        L.append("  QUÉ ARGUMENTO ATACA CADA RAZÓN: " + "; ".join(
            f"{r_['id']} ← {', '.join(comb.get(r_['id'], [])) or 'NINGUNO'}" for r_ in doc["razones"]))
    if doc.get("autonomas_sin_combatir"):
        L.append(f"  ⚠ RAZONES AUTÓNOMAS QUE NINGÚN ARGUMENTO COMBATE: {', '.join(doc['autonomas_sin_combatir'])}. "
                 f"Si una basta sola para sostener lo resuelto y nadie la ataca, derrotar las demás no "
                 f"cambia el resultado: la solución que prospere tiene que decir por qué no se sostiene.")
    if doc.get("hechos"):
        L.append("  HECHOS, CON SU CONDICIÓN:")
        for h in doc["hechos"]:
            L.append(f"   {h['id']} {h['que']} — {_TXT_COND.get(h['condicion'], h['condicion'])}"
                     + (f" ({h['fuente']}: «{h['cita'][:220]}»)" if h.get("cita") else ""))
    if doc.get("faltantes"):
        L.append("  FALTA EN LOS INSUMOS: " + "; ".join(f"{x['que']} ({x['por_que_importa']})" for x in doc["faltantes"]))
    L.append("  REGLA: distingue siempre «no consta en los insumos» de «no acreditado» y de «tenido por "
             "acreditado»; una razón autónoma no combatida sostiene lo resuelto; si falta un insumo "
             "indispensable, dilo en tu razón en vez de suplirlo.")
    return "\n".join(L) + "\n"
