# -*- coding: utf-8 -*-
"""LAS MARCAS DEL ESTUDIO — en qué párrafo contesta el estudio cada argumento.

POR QUÉ EXISTE. Plan del estudio de fondo aprobado por David el 26-sep-2026
(diag/w2_final.md §3.2, §4.4 V1 y §4.7). Con la v3 del prompt el estudio
recibe el inventario de argumentos (`inventario.py`) y escribe al PRINCIPIO del
párrafo que contesta uno o varios la marca con sus identificadores:
«⟦C1.a C1.b⟧». Con el plan (v4), además, «⟦M1⟧» donde expone una premisa y
«⟦U1⟧» en cada efecto de una unidad. Las marcas son registro interno:

  · NUNCA LLEGAN A LA PANTALLA. `FiltroMarcas` retiene el flujo desde «⟦» hasta
    «⟧» y quita la marca; la pantalla del secretario ve el texto escribiéndose
    sin ella. Y NUNCA SE TRAGA TEXTO: un «⟦» sin cierre en 200 caracteres se
    devuelve tal cual, igual que un corchete cuyo contenido no es una marca.
  · NUNCA LLEGAN AL .docx. `separar_marcas` las quita del texto final antes de
    componer (`redactor_adelanto._terminar`) y devuelve el MAPA: identificador
    → párrafos donde aparece.
  · `verificar` compara el mapa con el inventario: qué argumentos no tienen
    marca y, de ésos, cuáles tienen al menos RASTRO en el texto (sus anclas
    duras o sus palabras propias). Es el control V1 de la propuesta.

EL CONTROL V1 VA EN DOS NIVELES (w2_final §4.4, y la regla de la casa: todo
control que lee el texto generado se calibra contra engroses buenos antes de
volverse visible). Aquí no hay engroses que calibrar —los reales no llevan
marcas—, así que el aviso VISIBLE sólo sale cuando falta la marca Y el rescate
por anclas y texto tampoco encuentra el argumento. Lo demás va en sombra: se
registra en la ficha y en el evento «listo», no se enseña.

Por qué «⟦ ⟧» y no «[[ ]]»: los resúmenes llevan «[[p.7 §3]]», que el
compositor convierte en notas al pie. Una marca con los mismos corchetes se
confundiría con ellas. Aun así, si el modelo escribe la marca con corchetes
dobles —pasa cuando «normaliza» signos raros— se reconoce igual, pero SÓLO si
lo de dentro son identificadores: «[[p.7 §3]]» nunca lo es.
"""
from __future__ import annotations

import math
import re
from collections import Counter

ABRE, CIERRA = "⟦", "⟧"        # ⟦ ⟧
# Lo que se espera a que llegue el cierre. Una marca con veinte identificadores
# mide unos 120 caracteres; 200 deja margen sin retener de más.
LIMITE = 200

# C = concepto, A = agravio, AD = adhesivo, S = suplido (segmentos del
# inventario); M = premisa y U = unidad (del plan, v4).
_ID = r"(?:AD|[CASMU])\d{1,3}(?:\.[a-z]{1,2})?"
_RX_ID = re.compile(r"^(?:ad|[casmu])\d{1,3}(?:\.[a-z]{1,2})?$", re.I)
_RX_RANGO = re.compile(r"^((?:ad|[casmu])\d{1,3})\.([a-z])[-–—]\s*(?:(?:ad|[casmu])\d{1,3}\.)?([a-z])$", re.I)
# Las dos formas se buscan POR SEPARADO, primero «⟦…⟧» y luego «[[…]]»: con
# una sola expresión alternada, la coincidencia más a la izquierda —un
# «[[…]]» que no es marca— escondía una «⟦C1.a⟧» de dentro, y el texto final
# y el filtro del flujo no quitaban lo mismo (lo cazó la prueba de textos al
# azar de test_marcas.py).
_RX_MARCA_U = re.compile(ABRE + r"([^" + ABRE + CIERRA + r"\n]{1,%d})" % LIMITE + CIERRA)
_RX_MARCA_C = re.compile(r"\[\[([^\[\]\n]{1,%d})\]\]" % LIMITE)


def marcas_en(texto: str) -> list:
    """[(inicio, fin, ids)] de las marcas VÁLIDAS del texto, en las dos formas."""
    fuera = []
    for rx in (_RX_MARCA_U, _RX_MARCA_C):
        for m in rx.finditer(texto or ""):
            ids = ids_de(m.group(1))
            if ids:
                fuera.append((m.start(), m.end(), ids))
    return sorted(fuera)
_ENLACES = {"y", "e", "&"}


def _normal(tok: str) -> str:
    """«c1.A» → «C1.a»: el prefijo en mayúscula, la letra en minúscula."""
    m = re.match(r"^(ad|[casmu])(\d{1,3})(?:\.([a-z]{1,2}))?$", tok, re.I)
    if not m:
        return ""
    return m.group(1).upper() + m.group(2) + (f".{m.group(3).lower()}" if m.group(3) else "")


def ids_de(contenido: str) -> list:
    """Los identificadores de una marca, o [] si lo de dentro no es una marca.

    Admite separarlos con espacios, comas o punto y coma, una «y» antes del
    último y un rango «C1.a–C1.c». Si UN SOLO elemento no es identificador, no
    es una marca: se deja el texto como estaba."""
    fuera = []
    toks = [t for t in re.split(r"[\s,;·]+", (contenido or "").strip()) if t]
    if not toks:
        return []
    for t in toks:
        if t.lower() in _ENLACES:
            continue
        r = _RX_RANGO.match(t)
        if r:
            base, a, b = r.group(1), r.group(2).lower(), r.group(3).lower()
            if a > b:
                return []
            for c in range(ord(a), ord(b) + 1):
                fuera.append(_normal(f"{base}.{chr(c)}"))
            continue
        if not _RX_ID.match(t):
            return []
        fuera.append(_normal(t))
    vistos, unicos = set(), []
    for x in fuera:
        if x and x not in vistos:
            vistos.add(x)
            unicos.append(x)
    return unicos


def es_marca(contenido: str) -> bool:
    return bool(ids_de(contenido))


# ═══════════════════════════════════════════════════════════════════════════
# EN EL FLUJO
# ═══════════════════════════════════════════════════════════════════════════
class FiltroMarcas:
    """Quita las marcas del texto que se va escribiendo en pantalla.

    Retiene desde «⟦» (o «[[») hasta su cierre, porque la marca puede llegar
    partida entre dos trozos del flujo. Si lo de dentro son identificadores,
    la quita con UN blanco detrás —«⟦C1.a⟧ Sobre el primer…» sale «Sobre el
    primer…»—, aunque ese blanco llegue en el trozo siguiente. Si no, la
    devuelve tal cual. Un «⟦» sin cierre en `LIMITE` caracteres se devuelve
    tal cual: el filtro puede retrasar texto, nunca comérselo.

    Uso: `alimentar(trozo)` por cada trozo, y `cerrar()` al terminar el flujo
    para soltar lo que quedara retenido."""

    def __init__(self):
        self._pend = ""
        self._comer = False

    def alimentar(self, trozo: str) -> str:
        s = self._pend + (trozo or "")
        self._pend = ""
        out = []
        i, n = 0, len(s)
        while i < n:
            if self._comer:
                self._comer = False
                if s[i] in " \t":
                    i += 1
                    continue
            ka = s.find(ABRE, i)
            kb = s.find("[[", i)
            candidatos = [k for k in (ka, kb) if k >= 0]
            if not candidatos:
                # Un «[» al final puede ser la mitad de «[[»: se retiene uno.
                if s.endswith("[") and not s.endswith("[["):
                    out.append(s[i:n - 1])
                    self._pend = "["
                else:
                    out.append(s[i:])
                break
            k = min(candidatos)
            out.append(s[i:k])
            if s[k] == ABRE:
                abre, cierra = ABRE, CIERRA
            else:
                abre, cierra = "[[", "]]"
            m = s.find(cierra, k + len(abre))
            if m < 0:
                if n - k > LIMITE:
                    # Sin cierre a la vista: no es una marca. Tal cual.
                    out.append(abre)
                    i = k + len(abre)
                    continue
                self._pend = s[k:]
                break
            contenido = s[k + len(abre):m]
            if m - k > LIMITE or "\n" in contenido or not es_marca(contenido):
                out.append(abre)
                i = k + len(abre)
                continue
            i = m + len(cierra)
            self._comer = True
        return "".join(out)

    def cerrar(self) -> str:
        resto, self._pend, self._comer = self._pend, "", False
        return resto


# ═══════════════════════════════════════════════════════════════════════════
# EN EL TEXTO FINAL
# ═══════════════════════════════════════════════════════════════════════════
def parrafos(texto: str) -> list:
    """Los párrafos tal como los cuenta el mapa: renglones no vacíos."""
    return [x for x in (texto or "").split("\n") if x.strip()]


def separar_marcas(texto: str) -> tuple:
    """(texto sin marcas, mapa {«C1.a»: [índice de párrafo, …], «M1»: […]}).

    El índice cuenta los renglones no vacíos del texto limpio (`parrafos`).
    Una marca sola en su renglón vale para el párrafo siguiente, y el renglón
    desaparece. SIN MARCAS, EL TEXTO SALE IDÉNTICO, byte por byte: la v1 y la
    v2 pasan por aquí sin que les cambie nada."""
    texto = texto or ""
    if ABRE not in texto and "[[" not in texto:
        return texto, {}
    mapa = {}
    fuera = []
    idx = -1
    pendientes = []

    def _apunta(ids, k):
        for x in ids:
            lista = mapa.setdefault(x, [])
            if k not in lista:
                lista.append(k)

    for ln in texto.split("\n"):
        ids_ln = []

        def _sub(m):
            ids = ids_de(m.group(1))
            if not ids:
                return m.group(0)
            ids_ln.extend(ids)
            return "\x00"
        nueva = _RX_MARCA_C.sub(_sub, _RX_MARCA_U.sub(_sub, ln))
        if not ids_ln:
            if ln.strip():
                idx += 1
                _apunta(pendientes, idx)
                pendientes = []
            fuera.append(ln)
            continue
        nueva = re.sub(r"[ \t]*\x00[ \t]*", " ", nueva).strip()
        if nueva:
            idx += 1
            _apunta(pendientes + ids_ln, idx)
            pendientes = []
            fuera.append(nueva)
        else:
            pendientes.extend(ids_ln)
    if pendientes:
        # Marcas al final sin párrafo detrás: van al último que hubo.
        _apunta(pendientes, max(idx, 0))
    return "\n".join(fuera), mapa


def sin_marcas(texto: str) -> str:
    """El texto sin marcas (para los controles que leen el estudio)."""
    return separar_marcas(texto)[0]


# ═══════════════════════════════════════════════════════════════════════════
# EL CONTROL V1: ¿CADA ARGUMENTO TIENE SU MARCA?
# ═══════════════════════════════════════════════════════════════════════════
# EL RESCATE. Un argumento sin marca puede estar contestado igual: el modelo
# olvidó la marca, o lo contestó junto con otro y marcó sólo aquél. Antes de
# acusar se busca su RASTRO en el texto:
#   · un ancla dura PROPIA —la que no comparte con más de otro argumento del
#     inventario: «art 14 cpeum» sale en casi todos y no prueba nada— que
#     aparece en el estudio;
#   · o un párrafo que contiene una parte suficiente de las palabras de lo
#     que se alega —el TEXTO del segmento, no su cita: el estudio parafrasea
#     el argumento, no copia el escrito—, pesadas por lo raras que son en el
#     inventario.
# CALIBRADO SOBRE SALIDAS DEL SISTEMA, que es lo único que hay: los engroses
# reales no llevan marcas. Las 32 corridas v1/v2 del banco del 26-sep-2026 (8
# casos × 4, sin marcas: el peor caso, un estudio que no marcó nada) contra el
# localizador ciego, 452 pares segmento × corrida. Con 0.15 el rescate deja sin
# encontrar 4 de 347 segmentos que la corrida SÍ contestaba (1.2 %: la falsa
# alarma que vería el secretario) y 5 de 57 que sólo se cubrían con una
# fórmula genérica (9 %). Con la cita dentro del texto comparado, o con 0.20,
# la falsa alarma pasaba del 6 %. Un aviso que acusa a los inocentes enseña a
# no leer los avisos: se prefirió acusar poco y seguro.
UMBRAL_RASTRO = 0.15
MAX_COMPARTIDA = 2


def _raices(texto: str) -> set:
    import inventario as _inv
    return _inv._raices(texto)


def _anclas(texto: str) -> set:
    import inventario as _inv
    return set(_inv.anclas_duras(texto))


def rastros(segs: list, texto: str) -> dict:
    """{id: (por_ancla, parecido máximo con un párrafo)} de cada segmento."""
    if not segs:
        return {}
    ps = parrafos(texto)
    raiz_ps = [_raices(p) for p in ps]
    anclas_txt = _anclas(texto)
    por_seg = {s["id"]: _raices(str(s.get("texto") or "")) for s in segs}
    df = Counter(r for rs in por_seg.values() for r in rs)
    nseg = len(segs)
    idf = {r: math.log((nseg + 1) / (c + 0.5)) for r, c in df.items()}
    cuenta_ancla = Counter(a for s in segs for a in set(s.get("anclas") or []))
    fuera = {}
    for s in segs:
        propias = {a for a in (s.get("anclas") or []) if cuenta_ancla[a] <= MAX_COMPARTIDA}
        por_ancla = bool(propias & anclas_txt)
        rs = por_seg[s["id"]]
        tot = sum(idf[r] for r in rs) or 1.0
        mejor = 0.0
        for rp in raiz_ps:
            v = sum(idf[r] for r in rs & rp) / tot
            if v > mejor:
                mejor = v
        fuera[s["id"]] = (por_ancla, round(mejor, 3))
    return fuera


def verificar(segs: list, mapa: dict, texto_limpio: str) -> dict:
    """La cobertura por marcas y el rescate de las que faltan.

    {"faltan": ids sin marca, "rescatados": los de «faltan» con rastro en el
     texto, "sin_rastro": los que ni marca ni rastro —el aviso visible—,
     "cobertura": marcados / total, "cobertura_con_rescate": (marcados +
     rescatados) / total, "desconocidos": identificadores marcados que no
     están en el inventario (salvo M y U, que son del plan), "total",
     "marcados"}"""
    ids = [str(s.get("id")) for s in (segs or []) if s.get("id")]
    mapa = mapa or {}
    faltan = [i for i in ids if i not in mapa]
    rescatados, sin_rastro = [], []
    if faltan:
        # El peso de cada palabra se calcula sobre el inventario ENTERO: una
        # palabra que comparten diez argumentos no identifica a ninguno.
        r = rastros(segs, texto_limpio)
        for i in faltan:
            por_ancla, parecido = r.get(i, (False, 0.0))
            (rescatados if (por_ancla or parecido >= UMBRAL_RASTRO) else sin_rastro).append(i)
    conj = set(ids)
    desconocidos = sorted(k for k in mapa if k not in conj and not k.startswith(("M", "U")))
    total = len(ids)
    marcados = total - len(faltan)
    return {
        "faltan": faltan,
        "rescatados": rescatados,
        "sin_rastro": sin_rastro,
        "cobertura": round(marcados / total, 3) if total else 1.0,
        "cobertura_con_rescate": round((marcados + len(rescatados)) / total, 3) if total else 1.0,
        "desconocidos": desconocidos,
        "total": total,
        "marcados": marcados,
    }


def aviso(segs: list, cob: dict, q1: str = "concepto de violación") -> str:
    """El aviso VISIBLE: sólo los argumentos sin marca y sin rastro. Vacío si
    no hay ninguno. Nombra cada uno con lo que el resumen dice que se alega,
    para que el secretario lo busque sin abrir nada más."""
    faltan = list((cob or {}).get("sin_rastro") or [])
    if not faltan:
        return ""
    por_id = {s.get("id"): s for s in (segs or [])}
    partes = []
    for i in faltan[:6]:
        t = str((por_id.get(i) or {}).get("texto") or "")
        t = " ".join(t.split()[:22]) + ("…" if len(t.split()) > 22 else "")
        partes.append(f"{i} («{t}»)" if t else i)
    mas = f" y {len(faltan) - 6} más" if len(faltan) > 6 else ""
    return ("ARGUMENTOS SIN RESPUESTA IDENTIFICABLE: el estudio no marcó dónde "
            f"contesta {', '.join(partes)}{mas}, y tampoco se encontró su rastro "
            "en el texto. Revise que cada uno tenga su respuesta antes de firmar: "
            f"un argumento del {q1} sin contestar hace la sentencia inexhaustiva.")
