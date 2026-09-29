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

VERSION = "analisis-2"
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

# ═══ LA CITA CON ERRORES DE LECTURA ══════════════════════════════════════════
#
# Medido en el 103/2025: de 8 citas «sin verificar», las que se revisaron a
# mano estaban en el acto, pero el acto viene de un escaneo con errores de
# lectura («se derivan lidselementos constitutivos desus pretensiones»,
# «la senora pilar felipaz huefta») y el modelo citó el texto limpio. Con la
# comprobación literal, ese hecho bajaba a «sin_verificar» aunque el expediente
# lo dijera.
#
# LA TOLERANCIA ES DE LECTURA, NO DE SENTIDO, y se mide PALABRA POR PALABRA
# contra el tramo alineado (revisión adversarial: la primera versión medía
# letras y aceptaba «probó» donde el acto dice «no probó», «fundado» por
# «infundado» y «2025» por «2024»). Entre la cita y el tramo sólo se admite
# que un grupo corto de palabras se lea distinto con casi las mismas letras
# («los elementos» ↔ «lidselementos», «informado que la» ↔ «informad
# quezla»). Nunca: una palabra de más o de menos, una negación distinta, una
# cifra distinta, un prefijo o una palabra corta cambiada («y» ↔ «o»).
MIN_PALABRAS_OCR = 8
_CORTAS_CON_SENTIDO = frozenset(("y", "e", "o", "u", "si", "con", "mas", "menos", "ante", "sobre", "entre"))
_NEGACIONES = frozenset(("no", "ni", "sin", "nunca", "jamas", "tampoco", "nadie", "ninguno", "ninguna", "nada"))


def _lectura_distinta(qs: list, ss: list) -> bool:
    """¿Es `ss` (el texto) una mala lectura de `qs` (la cita)?"""
    import difflib
    if len(qs) > 3 or len(ss) > 3 or not qs or not ss:
        return False
    if any(w in _NEGACIONES for w in qs + ss):
        return False
    # «escritos o l as pruebas» contra «escritos y l as pruebas»: dentro de un
    # grupo pegado, la palabra corta que cambia el sentido tiene que ser la misma.
    if any(qs.count(w) != ss.count(w) for w in set(qs + ss) if w in _CORTAS_CON_SENTIDO):
        return False
    if any(ch.isdigit() for w in qs + ss for ch in w):
        return False
    a, b = "".join(qs), "".join(ss)
    if abs(len(a) - len(b)) > 2:
        return False
    if len(qs) == 1 and len(ss) == 1 and min(len(a), len(b)) <= 3:
        return False
    # Un prefijo puesto o quitado cambia el sentido («legal»/«ilegal»,
    # «procedente»/«improcedente»); una letra final perdida es lectura.
    if a != b and (a.endswith(b) or b.endswith(a)):
        return False
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio() >= 0.75


def _alinea(qw: list, win: list) -> bool:
    import difflib
    # a = la cita, b = el texto: «insert» es texto que la cita no trae (sólo
    # se tolera antes de su arranque y después de su final) y «delete» es
    # cita que el texto no trae (nunca).
    ops = difflib.SequenceMatcher(None, qw, win, autojunk=False).get_opcodes()
    while ops and ops[0][0] == "insert":
        ops.pop(0)
    while ops and ops[-1][0] == "insert":
        ops.pop()
    if not ops:
        return False
    iguales = sum(i2 - i1 for op, i1, i2, _, _ in ops if op == "equal")
    if iguales < 0.7 * len(qw):
        return False
    for op, i1, i2, j1, j2 in ops:
        if op == "equal":
            continue
        if op != "replace" or not _lectura_distinta(qw[i1:i2], win[j1:j2]):
            return False
    return True


def casi_literal(cita: str, texto) -> str:
    """La cita, si está en `texto` (un `plan_estudio.Texto`) salvo errores de
    lectura; si sobra alguna palabra en sus bordes, sin ellas (como
    `_recorte_literal`: sólo QUITA). «» si no está. Anclas de tres palabras
    votan el arranque del tramo."""
    import plan_estudio as _pe
    toks = str(cita or "").split()
    if len(_pe._palabras(cita)) < MIN_PALABRAS_OCR or not texto:
        return ""
    n = len(toks)
    minimo = max(MIN_PALABRAS_OCR, -(-3 * n // 4))
    candidatas = [toks] + [toks[a:n - (f - a)] for f in (1, 2, 3) for a in range(f + 1)]
    for plano in texto.planos:
        tw = plano.split()
        idx = getattr(texto, "_tejas", {}).get(id(plano))
        if idx is None:
            idx = {}
            for i in range(len(tw) - 2):
                idx.setdefault((tw[i], tw[i + 1], tw[i + 2]), []).append(i)
            if not hasattr(texto, "_tejas"):
                texto._tejas = {}
            texto._tejas[id(plano)] = idx
        for c in candidatas:
            if len(c) < minimo:
                continue
            qw = _pe._palabras(" ".join(c))
            votos = {}
            for j in range(len(qw) - 2):
                for p in idx.get((qw[j], qw[j + 1], qw[j + 2]), [])[:50]:
                    votos[p - j] = votos.get(p - j, 0) + 1
            # los arranques cercanos (±3) son el mismo tramo: se suman
            juntos = {}
            for s, v in votos.items():
                juntos[s // 4] = juntos.get(s // 4, 0) + v
            for g, v in sorted(juntos.items(), key=lambda kv: -kv[1])[:3]:
                if v < 2:
                    break
                a = max(0, g * 4 - 6)
                if _alinea(qw, tw[a:a + len(qw) + 12]):
                    return " ".join(c)
    return ""


def _lista(x) -> list:
    """Una lista aunque el modelo mande un valor suelto: «R1», «R1, R2», 2."""
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, str):
        return [s for s in re.split(r"[,;\s]+", x) if s]
    return [x]


def _relacion(x) -> str:
    """autonoma | conjunta | dependiente. Lo que no se entiende NO se da por
    autónomo: una razón autónoma sin atacar empuja a negar, y esa alarma sólo
    se da cuando el modelo lo dijo."""
    s = _plano_al(x)
    if s.startswith("autonom"):
        return "autonoma"
    if s.startswith("dependient"):
        return "dependiente"
    return "conjunta"


def _plano_al(x) -> str:
    import unicodedata
    s = unicodedata.normalize("NFD", str(x or "").lower())
    return " ".join("".join(ch for ch in s if unicodedata.category(ch) != "Mn").split())


def _fuente(x) -> str:
    s = _plano_al(x)
    if any(k in s for k in ("escrito", "agravio", "demanda", "concepto", "recurso")):
        return "escrito"
    if any(k in s for k in ("constancia", "autos", "prueba", "expediente")):
        return "constancias"
    return "acto"


def textos(acto: str = "", escrito: str = "", autos: str = "") -> dict:
    """Las tres fuentes listas para citar en ellas (`verificar_cita`)."""
    import plan_estudio as _pe
    return {"acto": _pe.Texto(acto or ""), "escrito": _pe.Texto(escrito or ""),
            "constancias": _pe.Texto(autos or "")}


def verificar_cita(c, T: dict, fuente: str = "acto") -> tuple:
    """(cita, verificada, fuente donde está, lectura «literal»|«ocr»|«»).

    LA MISMA REGLA PARA TODO EL TALLER (etapa 3): el análisis neutral, la
    justificación de cada solución y su revisión citan con esta función. Se
    busca primero en la fuente declarada y después en las demás —una fuente
    vacía nunca deja pasar la cita de otra con su nombre—; literal, recortada
    por los bordes (`_recorte_literal`, sólo quita) o, al final, salvo errores
    de lectura (`casi_literal`, calibrada: ver arriba)."""
    import plan_estudio as _pe
    c = _txt(c, 600)
    fuente = _fuente(fuente)
    if not c:
        return "", False, fuente, ""
    orden = [fuente] + [k for k in ("acto", "escrito", "constancias") if k != fuente]
    for k in orden:
        t = T.get(k)
        if not t:
            continue
        if t.contiene(c):
            return c, True, k, "literal"
        rec, _ = _pe._recorte_literal(c, [t])
        if rec:
            return rec, True, k, "literal"
    for k in orden:
        if T.get(k):
            rec = casi_literal(c, T[k])
            if rec:
                return rec, True, k, "ocr"
    return c, False, fuente, ""


def verificar(crudo: dict, acto: str, escrito: str, autos: str, segmentos: list) -> dict:
    """El análisis limpio: citas comprobadas, referencias válidas, condiciones
    del catálogo y las razones autónomas sin combatir calculadas por código.
    Nunca lanza por la forma de lo que devolvió el modelo."""
    T = textos(acto, escrito, autos)
    d = crudo if isinstance(crudo, dict) else {}

    ocr = set()

    def _cita(c, fuente: str) -> tuple:
        cita, ok, k, lectura = verificar_cita(c, T, fuente)
        if lectura == "ocr":
            ocr.add(cita)
        return cita, ok, k

    razones, ids_r = [], set()
    # Los ids que el modelo SÍ puso se reservan antes: la razón sin id no
    # puede quedarse con el de una que viene después (y borrarla).
    _explicitos = {_txt(x.get("id")).upper() for x in _lista(d.get("razones"))
                   if isinstance(x, dict) and _txt(x.get("id"))}
    for i, x in enumerate(_lista(d.get("razones")), 1):
        if not isinstance(x, dict):
            continue
        rid = _txt(x.get("id")).upper()
        if rid and rid in ids_r:
            continue
        if not rid:
            k = i
            while f"R{k}" in ids_r or f"R{k}" in _explicitos:
                k += 1
            rid = f"R{k}"
        ids_r.add(rid)
        cita, ok, _ = _cita(x.get("cita"), "acto")
        _probs = []
        for p in _lista(x.get("problemas")):
            try:
                _probs.append(int(str(p).strip()))
            except (TypeError, ValueError):
                pass
        razones.append({"id": rid, "afirma": _txt(x.get("afirma"), 800),
                        "conclusion": _txt(x.get("conclusion"), 600),
                        "relacion": _relacion(x.get("relacion")),
                        "con": [_txt(c).upper() for c in _lista(x.get("con")) if _txt(c)],
                        "problemas": _probs,
                        "cita": cita, "verificada": ok, **({"lectura": "ocr"} if cita in ocr else {})})
    for r_ in razones:
        r_["con"] = [c for c in r_["con"] if c in ids_r and c != r_["id"]]

    # LOS SEGMENTOS, POR SU ID SIN MAYÚSCULAS NI ESPACIOS; el mismo segmento
    # dos veces SUMA sus ataques (antes el segundo se perdía, y con él una
    # razón quedaba «sin combatir»).
    ids_seg = {_txt(s.get("id")).upper(): str(s.get("id")) for s in (segmentos or [])
               if isinstance(s, dict) and s.get("id")}
    argumentos, por_seg = [], {}
    for x in _lista(d.get("argumentos")):
        if not isinstance(x, dict):
            continue
        sid = _txt(x.get("segmento")).upper()
        if ids_seg:
            if sid not in ids_seg:
                continue
            sid = ids_seg[sid]
        combate = [c for c in (_txt(y).upper() for y in _lista(x.get("combate"))) if c in ids_r]
        if sid in por_seg:
            a = por_seg[sid]
            a["combate"] += [c for c in combate if c not in a["combate"]]
            continue
        por_seg[sid] = {"segmento": sid, "pide": _txt(x.get("pide"), 500),
                        "por_que": _txt(x.get("por_que"), 600), "combate": combate}
        argumentos.append(por_seg[sid])

    hechos, ids_h = [], set()
    for i, x in enumerate(_lista(d.get("hechos")), 1):
        if not isinstance(x, dict):
            continue
        fuente = _fuente(x.get("fuente"))
        cond = _txt(x.get("condicion")).lower()
        cond = cond if cond in CONDICIONES else "sin_verificar"
        cita, ok, fuente = _cita(x.get("cita"), fuente)
        # LA CONDICIÓN SE SOSTIENE CON SU FUENTE: «falta en insumos» no lleva
        # cita; si trae una que SÍ está en los insumos, la condición se
        # contradice y queda «sin_verificar» (el secretario la corrige).
        # Cualquier otra condición sin cita verificada no se da por buena.
        if cond == "falta_en_insumos":
            if cita and ok:
                cond = "sin_verificar"
            else:
                cita, ok = "", True
        elif not ok:
            cond = "sin_verificar"
        hid = _txt(x.get("id")).upper() or f"H{i}"
        if hid in ids_h:
            hid = f"H{i}"
            while hid in ids_h:
                hid += "b"
        ids_h.add(hid)
        hechos.append({"id": hid, "que": _txt(x.get("que"), 600),
                       "afirma": _txt(x.get("afirma"), 60), "fuente": fuente,
                       "cita": cita, "verificada": ok, "condicion": cond,
                       **({"lectura": "ocr"} if cita and cita in ocr else {})})

    faltantes = [{"que": _txt(x.get("que"), 300), "por_que_importa": _txt(x.get("por_que_importa"), 400)}
                 for x in _lista(d.get("faltantes")) if isinstance(x, dict) and _txt(x.get("que"))]

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


MAX_RAZONES_BLOQUE = 20
MAX_HECHOS_BLOQUE = 30


def bloque_propuesta(doc: dict | None) -> str:
    """El análisis como DATOS para la propuesta, con su regla. «» si no hay."""
    if not isinstance(doc, dict) or not (doc.get("razones") or doc.get("hechos")):
        return ""
    L = ["", "ANÁLISIS NEUTRAL DE LA LITIS (datos previos a decidir; no califica):"]
    if doc.get("cuestion_central"):
        L.append(f"  CUESTIÓN CENTRAL: {doc['cuestion_central']}")
    if doc.get("razones"):
        L.append("  RAZONES DE LO RESUELTO:")
        for r_ in doc["razones"][:MAX_RAZONES_BLOQUE]:
            rel = {"autonoma": "AUTÓNOMA (basta sola)", "conjunta": "CONJUNTA",
                   "dependiente": "DEPENDIENTE"}.get(r_.get("relacion"), "CONJUNTA")
            con = f" con {', '.join(r_['con'])}" if r_.get("con") else ""
            L.append(f"   {r_['id']} [{rel}{con}] {r_['afirma']} → {r_['conclusion']}"
                     + (f"\n       cita{'' if r_['verificada'] else ' NO verificada'}"
                        f"{' (cotejada salvo errores de lectura del texto: no es copia letra por letra)' if r_.get('lectura') == 'ocr' else ''}"
                        f": «{r_['cita']}»" if r_.get("cita") else ""))
    comb = {}
    for a in doc.get("argumentos") or []:
        for c in a.get("combate") or []:
            comb.setdefault(c, []).append(a["segmento"])
    if doc.get("razones"):
        L.append("  QUÉ ARGUMENTO ATACA CADA RAZÓN: " + "; ".join(
            f"{r_['id']} ← {', '.join(comb.get(r_['id'], [])) or 'NINGUNO'}" for r_ in doc["razones"][:MAX_RAZONES_BLOQUE]))
    if doc.get("autonomas_sin_combatir"):
        L.append(f"  ⚠ RAZONES AUTÓNOMAS QUE NINGÚN ARGUMENTO COMBATE: {', '.join(doc['autonomas_sin_combatir'])}. "
                 f"Si una basta sola para sostener lo resuelto y nadie la ataca, derrotar las demás no "
                 f"cambia el resultado: la solución que prospere tiene que decir por qué no se sostiene.")
    if doc.get("hechos"):
        L.append("  HECHOS, CON SU CONDICIÓN:")
        for h in doc["hechos"][:MAX_HECHOS_BLOQUE]:
            L.append(f"   {h['id']} {h['que']} — {_TXT_COND.get(h.get('condicion'), h.get('condicion'))}"
                     + (f" ({h.get('fuente')}{', salvo errores de lectura' if h.get('lectura') == 'ocr' else ''}"
                        f": «{str(h['cita'])[:220]}»)" if h.get("cita") else ""))
    if doc.get("faltantes"):
        L.append("  FALTA EN LOS INSUMOS: " + "; ".join(f"{x.get('que')} ({x.get('por_que_importa')})"
                                                     for x in doc["faltantes"][:12]))
    L.append("  REGLA: distingue siempre «no consta en los insumos» de «no acreditado» y de «tenido por "
             "acreditado»; una razón autónoma no combatida sostiene lo resuelto; si falta un insumo "
             "indispensable, dilo en tu razón en vez de suplirlo.")
    return "\n".join(L) + "\n"
