#!/usr/bin/env python3
"""EL ÍNDICE DE VIGENCIA DE LAS TESIS → datos/vigencia_tesis.json.   (25-sep-2026)

POR QUÉ EXISTE
--------------
David le preguntó al chat si los Tribunales Colegiados pueden ejercer control
difuso sobre normas del juicio de origen, y el chat le dio como vigentes la
P. X/2015 (10a.) (registro 2009817) y la P. IX/2015 (2009816). Las dos están
ABANDONADAS desde febrero de 2022 por la P./J. 2/2022 (11a.) (2024159). La nota
de abandono estaba en nuestro acervo —en `payload["precedentes"]` de
`jurisprudencia_nacional_v3`— pero ninguna ruta del chat la leía: al modelo le
llegaban la cabecera y el rubro, y el sello final dio por buenas las dos tesis
porque su registro existe. La sustituta estaba en el puesto 38 del vector de la
abandonada: por similitud nunca iba a llegar.

Este guion lee esas notas UNA vez, fuera de la consulta, y deja un índice
pequeño (558 tesis, ~0.4 MB) que `vigencia_tesis.py` carga en memoria. Es la
versión de producción del prototipo del 25-sep-2026 (pilar/vigencia_prototipo.py):
las reglas son las mismas, medidas contra el Semanario —40 de 40 en una muestra
estratificada por estado y fuente (cota inferior de Wilson al 95 %: 91 %), más
11 de 11 en las que salieron del SJF y los casos 2009817/2009816—.

QUÉ LEE (sólo lectura; nada de embeddings ni modelos)
-----------------------------------------------------
  · Qdrant, `jurisprudencia_nacional_v3`: registro, clave, rubro, precedentes,
    tipo, época, fecha de publicación. Scroll, sin vectores.
  · La caché del SJF (`--sjf-cache`) para las 1,370 tesis cuyo `precedentes`
    quedó CORTADO a 2,500 caracteres en la ingesta v3_semanario_2026-08: las
    notas de vigencia van al final y el corte se las comía. 2024159 perdió así su
    propia nota («La presente tesis abandona… P. IX/2015 y P. X/2015»). Con la
    caché salieron 4 afectadas nuevas y cambiaron 4 reemplazos. Sin la caché el
    guion corre igual y lo dice (esas tesis quedan como «qdrant_truncado»).

DE DÓNDE SALE CADA PÉRDIDA (con la dirección explícita)
-------------------------------------------------------
  · nota_propia : la tesis afectada lo dice de sí misma
                  («La presente tesis fue abandonada por…», «Este criterio fue
                  interrumpido por…», «…abandonó el criterio sostenido en esta
                  tesis…», «…dejó de considerarse de aplicación obligatoria…»).
  · tesis_nueva : la tesis nueva declara lo que reemplaza (rubro «[ABANDONO DE
                  LA TESIS …]», «(INTERRUPCIÓN …)», «(SUSTITUCIÓN …)»; notas
                  «Esta tesis interrumpe…», «Tesis sustituida:», «ya no se
                  considera de aplicación obligatoria la diversa…»).
  · cita_tercera: una tercera tesis relata la pérdida («Han quedado sin efectos
                  las tesis jurisprudenciales P./J. 73/99 y…»).

LO QUE NO ES PÉRDIDA y no se cuenta: «declaró inexistente / sin materia /
improcedente la contradicción», «contendió en la contradicción», los votos
concurrentes que «se apartan del criterio» (15 casos), las notas de norma
reformada («el artículo … a que se hace referencia en esta tesis fue modificado
mediante decreto») y las republicaciones que sólo corrigen el texto. Las guardas
que los descartan están abajo, cada una con el caso que la motivó.

LO QUE LAS REGLAS NO VEN
------------------------
La superación implícita, sin nota en el Semanario (160584, P. LXVI/2011, frente a
la P./J. 21/2014): ésa va en la capa curada `datos/vigencia_curada.json`, que
este guion NO toca.

PENDIENTE (a propósito fuera del índice)
----------------------------------------
  · Las 6,282 tesis con «ver jurisprudencia» —contendieron en una contradicción
    (6,001), o ésta se declaró sin materia (632), improcedente (340) o
    inexistente (34)— NO perdieron vigencia: son otra cosa, un aviso suave
    («contendió en la CT X; ver la jurisprudencia Y»), nunca «abandonada».
    `--informes DIR` las deja en DIR/vigencia_ver_jurisprudencia.json para cuando
    se decida cómo mostrarlas.
  · La fecha «desde» es la publicación del reemplazo (o la de la resolución que
    la nota dé); el lunes de obligatoriedad sólo sale si la nota lo dice: el SJF
    lo trae en `textoPublicacion`, que Qdrant no guarda (2009817: el índice dice
    11-feb-2022, el SJF «obligatoria desde el lunes 14»).
  · 12 claves ambiguas (misma clave con otro rubro) y 27 afectadas fuera del
    acervo (sobre todo de la Octava) quedan sin registro: van a
    DIR/vigencia_sin_resolver.json.
  · El arreglo de fondo es la ingesta: guardar `precedentes` completo y correr
    esto cada semana con la carga del Semanario. El SJF añade notas a tesis
    viejas: 2019978 y 2029850 se abandonaron el 12-ago-2026.

USO
---
    .venv/bin/python scripts/vigencia_tesis_generar.py \\
        --env ../../../.env --sjf-cache <dir> [--salida datos/vigencia_tesis.json] \\
        [--tesis-cache tesis.jsonl] [--informes DIR]

`--tesis-cache` guarda (o reutiliza, si existe) el volcado de las 71,655 tesis
para no releer Qdrant en cada ensayo. Tarda ~20 s con la caché.
"""
from __future__ import annotations

import argparse
import collections
import datetime as _dt
import difflib
import json
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

RAIZ = Path(__file__).resolve().parents[1]
COLECCION = "jurisprudencia_nacional_v3"
CAMPOS = ["registro", "clave_tesis", "rubro", "precedentes", "tipo", "epoca",
          "fecha_publicacion", "instancia", "materia"]
SALIDA = RAIZ / "datos" / "vigencia_tesis.json"
SJF_CACHE_OMISION = ("/private/tmp/claude-501/-Users-josedavidalcantarmendoza-Documents-IUREXIA-MAC-"
                     "jurexia-api-git--claude-worktrees-xenodochial-poincare-fb5468/"
                     "ef3b257b-8e89-453a-8ee1-4c9ffa7ff848/scratchpad/pilar/sjf_cache")

# Topes del archivo compacto: el índice viaja con el código y se carga en cada
# arranque del servidor. La nota se guarda LITERAL (es lo que se le muestra al
# modelo como prueba), cortada en palabra entera.
TOPE_NOTA = 400
# Lo que se recorta del párrafo ANTES de cortar en palabra entera: el prototipo
# cortaba a 400 a pelo y luego no había de dónde completar la palabra.
NOTA_BRUTA = 600
TOPE_RUBRO = 200


# --------------------------------------------------------------------------- carga
def _env(ruta: Path) -> dict:
    """QDRANT_URL y QDRANT_API_KEY del .env (o del entorno). No se imprimen."""
    e = {k: os.environ[k] for k in ("QDRANT_URL", "QDRANT_API_KEY") if os.environ.get(k)}
    if ruta and ruta.exists():
        for l in ruta.read_text(encoding="utf-8").splitlines():
            if "=" in l and not l.lstrip().startswith("#"):
                k, v = l.split("=", 1)
                e.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    return e


def cargar(env_ruta: Path, tesis_cache: Optional[Path] = None) -> List[dict]:
    """Las tesis de la v3 con los campos que miran las reglas. Sólo lectura."""
    if tesis_cache and tesis_cache.exists():
        with tesis_cache.open(encoding="utf-8") as fh:
            return [json.loads(l) for l in fh]
    e = _env(env_ruta)
    if not e.get("QDRANT_URL"):
        sys.exit(f"Sin QDRANT_URL (ni en el entorno ni en {env_ruta}).")
    from qdrant_client import QdrantClient
    q = QdrantClient(url=e["QDRANT_URL"], api_key=e.get("QDRANT_API_KEY"), timeout=120)
    out, off = [], None
    while True:
        pts, off = q.scroll(COLECCION, limit=2000, offset=off, with_payload=CAMPOS, with_vectors=False)
        for p in pts:
            d = dict(p.payload or {})
            d["_id"] = str(p.id)
            out.append(d)
        if off is None:
            break
    if tesis_cache:
        with tesis_cache.open("w", encoding="utf-8") as fh:
            for d in out:
                fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    return out


# --------------------------------------------------------------------------- precedentes completos (SJF)
SJF_CACHE = SJF_CACHE_OMISION     # se fija desde --sjf-cache en main()
TOPE_QDRANT = 2500  # jurisprudencia_nacional_v3 guarda `precedentes` cortado en 2,500 caracteres


def precedentes_completos(t: dict) -> Tuple[str, str]:
    """Si el precedente de Qdrant llegó cortado y hay copia del SJF en la caché local, usa la del SJF
    (las notas de vigencia van al final y el corte se las come). -> (texto, origen)."""
    prec = t.get("precedentes") or ""
    if len(prec) < TOPE_QDRANT:
        return prec, "qdrant"
    ruta = os.path.join(SJF_CACHE, f"{t.get('registro')}.json")
    if not os.path.exists(ruta):
        return prec, "qdrant_truncado"
    try:
        with open(ruta, encoding="utf-8") as fh:
            d = json.load(fh)
    except Exception:
        return prec, "qdrant_truncado"
    h = d.get("precedentes") or ""
    h = re.sub(r"</p>\s*(?:<br\s*/?>\s*)*<p[^>]*>", "\n\n", h)
    h = re.sub(r"<br\s*/?>", "\n", h)
    h = re.sub(r"<[^>]+>", "", h)
    h = h.replace("&nbsp;", " ").replace("&quot;", '"').replace("&amp;", "&")
    return (h, "sjf") if len(h) > len(prec) else (prec, "qdrant_truncado")


# --------------------------------------------------------------------------- normalización
def plegar(s: str) -> str:
    s = unicodedata.normalize("NFD", s or "")
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


SUF_RX = re.compile(r"\(\s*(9|10|11|12)\s*a\s*\.?\s*\)", re.I)


def norm_clave(c: str) -> Tuple[str, Optional[str]]:
    """'P. X/2015 (10a.)' -> ('P.X/2015', '10').  Quita espacios, (*), acentos."""
    c = plegar(c or "").upper().replace("º", "O").replace("°", "O")
    c = c.replace("(*)", "")
    m = SUF_RX.search(c)
    suf = m.group(1) if m else None
    base = SUF_RX.sub("", c)
    base = re.sub(r"\s+", "", base).strip(" .,;:")
    base = re.sub(r"^(P|1A|2A|3A|4A)\.J\.", r"\1./J.", base)   # errata sin diagonal
    return base, suf


def norm_rubro(r: str) -> str:
    r = plegar(r or "").upper()
    r = re.sub(r"<[^>]+>", " ", r)
    r = re.sub(r"[^A-Z0-9 ]+", " ", r)
    return re.sub(r"\s+", " ", r).strip()


# Candidatos a clave de tesis dentro de un texto. Se validan contra el índice, así que
# puede ser generoso.
_SUF = r"(?:\s*\(\s*(?:9|10|11|12)a\.\s*\))?(?:\s*\(\*\))?"
_REG = r"\(\s*[IVXL]+\s*Regi[óo]n\s*\)\s*"
CLAVE_RX = re.compile(
    r"(?<![\w./])(?:"
    r"(?:P|1a|2a|3a|4a)\.\s*(?:/?\s*J\.\s*)?(?:\d+|[IVXLCDM]+)\s*/\s*\d{2,4}"          # SCJN (tolera "1a. J. 96/2011")
    r"|(?:PC|PR)\.[A-Z0-9.\s]*?(?:J/\s*\d+|\d+)\s*[A-Z]{1,2}\b"                          # Plenos
    r"|(?:" + _REG + r")?[IVXL]+\.(?:\d+o\.)?(?:" + _REG + r")?(?:[A-Z]{1,2}\.)*\s*"
    r"(?:J/\s*\d+(?:\s*[A-Z]{1,2}\b)?|\d+\s*[A-Z]{1,2}\b)"                               # TCC
    r"|" + _REG + r"\d+o\.(?:[A-Z]{1,2}\.)*\s*(?:J/\s*\d+(?:\s*[A-Z]{1,2}\b)?|\d+\s*[A-Z]{1,2}\b)"  # auxiliares
    r")" + _SUF
)
REGISTRO_RX = re.compile(r"registro(?:s)?\s+digital(?:es)?\s*:?\s*(\d{6,7})", re.I)
RUBRO_Q_RX = re.compile(
    r"(?:rubros?|t[íi]tulo y subt[íi]tulo|subt[íi]tulo)\s*:?\s*[\"“«]([^\"”»]{12,1800}?)[\"”»]", re.I)

MESES = {m: i for i, m in enumerate(
    ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
     "septiembre", "octubre", "noviembre", "diciembre"], 1)}
FECHA_RX = re.compile(r"(\d{1,2})\s+de\s+(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|"
                      r"octubre|noviembre|diciembre)\s+de\s+(\d{4})", re.I)
MES_ANIO_RX = re.compile(r"(?:Tomo|Libro|N[úu]mero)[^,]{0,40},(?:\s*Tomo[^,]{0,20},)?(?:\s*Volumen[^,]{0,10},)?\s*"
                         r"(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)"
                         r"\s+(?:de\s+)?(\d{4})", re.I)


_UNI = {"un": 1, "uno": 1, "primero": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7,
        "ocho": 8, "nueve": 9, "diez": 10, "once": 11, "doce": 12, "trece": 13, "catorce": 14, "quince": 15,
        "dieciseis": 16, "diecisiete": 17, "dieciocho": 18, "diecinueve": 19, "veinte": 20, "veintiun": 21,
        "veintiuno": 21, "veintidos": 22, "veintitres": 23, "veinticuatro": 24, "veinticinco": 25,
        "veintiseis": 26, "veintisiete": 27, "veintiocho": 28, "veintinueve": 29, "treinta": 30, "cuarenta": 40,
        "cincuenta": 50, "sesenta": 60, "setenta": 70, "ochenta": 80, "noventa": 90}


def numero_palabras(t: str) -> Optional[int]:
    """'dos mil cinco' -> 2005; 'mil novecientos noventa y nueve' -> 1999; 'treinta y uno' -> 31."""
    t = plegar(t).lower().split()
    total, acc = 0, 0
    for w in t:
        if w == "y":
            continue
        if w == "mil":
            total += (acc or 1) * 1000
            acc = 0
        elif w == "novecientos":
            acc += 900
        elif w in _UNI:
            acc += _UNI[w]
        else:
            return None
    return total + acc


FECHA_PALABRAS_RX = re.compile(
    r"\b([a-záéíóú]+(?: y [a-záéíóú]+)?) de (enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|"
    r"noviembre|diciembre) de ((?:dos mil|mil novecientos)(?: [a-záéíóú]+){0,3})", re.I)


def fecha_palabras(p: str) -> Optional[str]:
    for m in FECHA_PALABRAS_RX.finditer(p):
        d = numero_palabras(m.group(1))
        ws = m.group(3).split()
        a = None
        while ws and a is None:
            a = numero_palabras(" ".join(ws))
            ws = ws[:-1]
        if d and a and 1 <= d <= 31 and 1900 < a < 2100:
            return f"{a:04d}-{MESES[m.group(2).lower()]:02d}-{d:02d}"
    return None


def iso(d: str, m: str, a: str) -> str:
    return f"{int(a):04d}-{MESES[m.lower()]:02d}-{int(d):02d}"


# --------------------------------------------------------------------------- índices
class Acervo:
    def __init__(self, tesis: List[dict]):
        self.por_reg: Dict[str, dict] = {}
        self.por_base: Dict[str, List[Tuple[Optional[str], str]]] = collections.defaultdict(list)
        self.por_rubro: Dict[str, List[str]] = collections.defaultdict(list)
        self.por_rubro80: Dict[str, List[str]] = collections.defaultdict(list)
        for t in tesis:
            r = str(t.get("registro"))
            self.por_reg[r] = t
            c = (t.get("clave_tesis") or "").strip()
            if c:
                b, s = norm_clave(c)
                self.por_base[b].append((s, r))
            nr = norm_rubro(t.get("rubro") or "")
            if nr:
                self.por_rubro[nr].append(r)
                self.por_rubro80[nr[:80]].append(r)

    def clave_de(self, r: str) -> Tuple[str, Optional[str]]:
        return norm_clave(self.por_reg.get(r, {}).get("clave_tesis") or "")

    def resolver_clave(self, texto: str, excluir: Optional[str] = None,
                       rubro_hint: Optional[str] = None) -> Tuple[Optional[str], str]:
        """-> (registro | None, confianza 'alta'|'media'|'ambigua'|'no_en_acervo')."""
        self.ultimos_duplicados = []
        b, s = norm_clave(texto)
        cands = [(cs, r) for cs, r in self.por_base.get(b, []) if r != excluir]
        if not cands:
            return None, "no_en_acervo"
        elegidos: List[str]
        if s:
            elegidos = [r for cs, r in cands if cs == s]
            conf = "alta"
            if not elegidos:
                elegidos = [r for cs, r in cands if cs is None]
                conf = "media"
        else:
            elegidos = [r for cs, r in cands if cs is None]
            conf = "alta"
            if not elegidos:
                elegidos = [r for _, r in cands]
                conf = "media"
        if len(elegidos) > 1 and rubro_hint:
            nr = norm_rubro(rubro_hint)
            fil = [r for r in elegidos if norm_rubro(self.por_reg[r].get("rubro") or "")[:60] == nr[:60]]
            if fil:
                elegidos = fil
        if len(elegidos) == 1:
            return elegidos[0], conf
        if elegidos and len({norm_rubro(self.por_reg[r].get("rubro") or "")[:60] for r in elegidos}) == 1:
            # misma clave y mismo rubro: es la misma tesis publicada dos veces (republicación,
            # aclaración). Se devuelve la más reciente; las demás en self.ultimos_duplicados.
            orden = sorted(elegidos, key=lambda r: orden_registro(r) or 0)
            self.ultimos_duplicados = orden[:-1]
            return orden[-1], conf
        return None, "ambigua" if elegidos else "no_en_acervo"

    ultimos_duplicados: List[str] = []

    def resolver_rubro(self, rubro: str, excluir: Optional[str] = None) -> Optional[str]:
        nr = norm_rubro(rubro)
        if len(nr) < 25:
            return None
        c = [r for r in self.por_rubro.get(nr, []) if r != excluir]
        if len(c) == 1:
            return c[0]
        c = [r for r in self.por_rubro80.get(nr[:80], []) if r != excluir]
        return c[0] if len(c) == 1 else None

    def rubro_coincide(self, r: str, rubro: Optional[str]) -> Optional[bool]:
        if not rubro or r not in self.por_reg:
            return None
        a = norm_rubro(self.por_reg[r].get("rubro") or "")
        b = norm_rubro(rubro)
        if not a or not b:
            return None
        n = min(len(a), len(b), 70)
        if a[:n] == b[:n]:
            return True
        # erratas del propio Semanario ("CONTA" por "CONTRA"): se tolera una similitud alta
        return difflib.SequenceMatcher(None, a[:n], b[:n]).ratio() >= 0.9


# --------------------------------------------------------------------------- utilidades de texto
def parrafos(prec: str) -> List[str]:
    prec = re.sub(r"<[^>]+>", " ", prec or "")
    partes = re.split(r"\n\s*\n|\n(?=\s*Nota\s*:)", prec)
    return [re.sub(r"[ \t]+", " ", p).strip() for p in partes if p and p.strip()]


def claves_pos(texto: str) -> List[Tuple[str, int, int]]:
    return [(m.group(0).strip(), m.start(), m.end()) for m in CLAVE_RX.finditer(texto)]


def claves_en(texto: str) -> List[str]:
    out = []
    for m in CLAVE_RX.finditer(texto):
        c = m.group(0).strip()
        if c not in out:
            out.append(c)
    return out


def orden_registro(r: str) -> Optional[float]:
    """Orden cronológico aproximado del registro digital: el de la Novena desciende con el
    tiempo (199xxx en 1995 … 160xxx en 2011); desde 2000000 asciende."""
    try:
        n = int(r)
    except (TypeError, ValueError):
        return None
    if n >= 2000000:
        return 1e7 + n
    return 1e6 - n


def fecha_pub(t: Optional[dict]) -> Optional[str]:
    f = (t or {}).get("fecha_publicacion") or ""
    m = re.match(r"(\d{4}-\d{2}-\d{2})", f)
    return m.group(1) if m else None


def recorte(p: str, i: int) -> str:
    """Hasta NOTA_BRUTA caracteres literales del párrafo, desde el inicio de la oración que contiene i."""
    j = p.rfind(". ", 0, max(0, i - 1)) if i > 0 else -1
    ini = j + 2 if j >= 0 else 0
    if i - ini > 250:
        ini = max(0, i - 120)
    return p[ini:ini + NOTA_BRUTA].strip()


def registro_tras(p: str, i: int, fin: Optional[int] = None) -> Optional[str]:
    """Primer 'registro digital: N' entre la posición i y fin (o 900 caracteres)."""
    fin = fin if fin is not None else i + 900
    m = REGISTRO_RX.search(p, i, min(len(p), fin))
    return m.group(1) if m else None


# --------------------------------------------------------------------------- patrones
V_ESTADO = [
    (r"abandon|apart", "abandonada"),
    (r"interrump|interrupc", "interrumpida"),
    (r"sustitu", "sustituida"),
    (r"superad|supera", "superada"),
    (r"modific", "modificada"),
    (r"cancel|sin efecto", "sin_efectos"),
    (r"aclar", "aclarada"),
    (r"acot", "modificada"),
]


def estado_de(v: str) -> str:
    v = plegar(v or "").lower()
    for rx, e in V_ESTADO:
        if re.search(rx, v):
            return e
    return "superada"


# (a) la tesis habla de sí misma: ella es la afectada.
PROPIA = [
    ("tesis_fue", re.compile(
        r"(?:la presente|esta|dicha) (?:tesis|jurisprudencia)(?: jurisprudencial| aislada| de jurisprudencia)?,?\s+(?:fue|ha sido|qued[óo])\s+"
        r"(?P<parc>parcialmente\s+)?(?P<v>abandonad|interrumpid|sustituid|superad|cancelad|modificad)[ao](?!\s+para que guardara)"
        r"(?P<parc2>\s+parcialmente)?", re.I)),
    ("criterio_fue", re.compile(
        r"(?:este criterio|(?:el\s+)?criterio (?:contenido|sostenido|sustentado) en (?:la presente|esta) (?:tesis|jurisprudencia))"
        r",?\s+(?:fue|ha sido|qued[óo]|se encuentra)\s+(?P<parc>parcialmente\s+)?"
        r"(?P<v>abandonad|interrumpid|superad|sustituid|modificad)[oa](?P<parc2>\s+parcialmente)?", re.I)),
    ("abandono_el_criterio_de_esta", re.compile(
        r"(?P<v>abandon[óo]|abandonan?|abandonaron|se apart[óo]|se apartan?|se apartaron|interrumpi[óo]|interrumpe|"
        r"modific[óo]|acotan?|acot[óo]|"
        r"determin[óo] (?:modificar|abandonar|sustituir|interrumpir|dejar sin efectos))"
        r"(?P<parc>\s+parcialmente)?,?[^.;]{0,90}?\s(?:del?|el|los) criterios?\s+(?:sostenidos?|contenidos?|sustentados?)\s+"
        r"(?:en|por)\s+(?:esta|la presente)\s+(?:tesis|jurisprudencia)", re.I)),
    ("dejo_de_ser_obligatoria", re.compile(
        r"(?:(?:la presente|esta) tesis|[ée]sta(?!\s+[úu]ltima))\s+dej[óo] de considerarse de aplicaci[óo]n obligatoria", re.I)),
    ("cancelada", re.compile(r"(?P<v>orden[óo] cancelar) (?:la presente|esta) tesis|(?:la presente|esta) tesis "
                             r"(?P<v2>se cancel[óo])|(?:la presente|esta) tesis(?:,? que aparece publicada)?[^.]{0,220}?,?\s+"
                             r"(?P<v3>se cancela) por instrucciones|(?:la presente|esta) tesis (?:fue|ha sido) (?P<v4>cancelada)", re.I)),
    ("aclarada", re.compile(
        r"(?:el\s+)?criterio (?:contenido|sostenido|sustentado) en (?:la presente|esta) tesis(?: aislada| jurisprudencial)?\s+"
        r"se (?P<v>aclar[óo]) para quedar", re.I)),
]

# (b) la tesis declara lo que reemplaza: ella es la nueva.
NUEVA = [
    ("esta_tesis_abandona", re.compile(
        r"(?:en\s+)?(?:la presente|esta) tesis(?: jurisprudencial| aislada| de jurisprudencia)?,?\s+(?:se\s+)?"
        r"(?P<v>abandona|interrumpe|se aparta|modifica|sustituye|supera)(?P<parc>\s+parcialmente|,\s+en lo conducente,)?\s+"
        r"(?:(?:el|los|del?)\s+criterios?|(?:a\s+)?(?:la|las)\s+(?:diversas?|tesis|jurisprudencias?))", re.I)),
    ("en_virtud_de_que", re.compile(
        r"en virtud de que\s+(?P<v>abandona|interrumpe|sustituye|supera|modifica|se aparta)(?P<parc>\s+parcialmente)?\s+"
        r"(?:(?:el|los|del?)\s+criterios?|(?:la|las)\s+(?:diversas?|tesis|jurisprudencias?))", re.I)),
    ("ya_no_obligatoria", re.compile(
        r"(?P<v>ya no se consideran? de aplicaci[óo]n obligatoria)\s+(?:la|las)\s+"
        r"(?:diversas?|tesis|jurisprudencias?)", re.I)),
    ("deriva_de_modificacion", re.compile(
        r"(?:la presente|esta) tesis deriva de la resoluci[óo]n dictada en (?:la|el) (?:solicitud|expediente).{0,700}?"
        r"determin[óo] (?P<v>modificar|abandonar|sustituir|interrumpir|dejar sin efectos?) (?:el|los|las|la) "
        r"(?:criterios? contenidos? en (?:la|las) )?(?:tesis|jurisprudencias?)(?: jurisprudenciales| de jurisprudencia)?"
        r"(?: n[úu]meros?)?", re.I)),
    ("rubro_y_texto_sustituyen", re.compile(
        r"(?:el\s+)?(?:rubro y\s+)?texto (?:de|contenido en) (?:la presente|esta) tesis,?\s+(?P<v>sustituyen?)\s+a\s+"
        r"(?:los de\s+|el de\s+)?(?:la|el)\s+(?:tesis|diversa)(?:\s+n[úu]mero)?", re.I)),
    ("tesis_sustituye_a_la", re.compile(
        r"(?:la presente|esta) tesis(?: jurisprudencial)?\s+(?P<v>sustituye)\s+a\s+la\s+que\s+con\s+el\s+n[úu]mero", re.I)),
    ("texto_se_sustituye", re.compile(
        r"el texto de la tesis\s+(?P<c>[^,]{3,40}),[^.]{0,200}?(?P<v>se sustituye) por el de [ée]sta", re.I)),
]

TESIS_SUSTITUIDA_RX = re.compile(
    r"(?:Tesis|Jurisprudencias?|Criterios?)(?: y/o criterios?)? (?:sustituid[ao]s?|que se sustituyen?)\s*:\s*(?P<cuerpo>.{0,1800})",
    re.I | re.S)

# (c) lo relata una tercera tesis (o la propia, con su clave).
SIN_EFECTOS_RX = re.compile(
    r"(?:[Hh]an|[Hh]a) quedado sin efectos?,?\s+(?:las|la)\s+(?:tesis\s+)?(?:jurisprudenciales|jurisprudencias?|tesis)"
    r"(?:\s+n[úu]meros?)?\s+(?P<lista>[^:;]{3,160}?)(?:,\s*(?:cuyos|de)\s+rubros?|[:;]|$)")
SIN_EFECTOS_B_RX = re.compile(
    r"(?:Las|La)\s+(?:tesis\s+)?(?:jurisprudencias?|tesis)(?:\s+de jurisprudencia)?\s+(?P<lista>[^:;]{3,120}?),?\s+"
    r"(?:han|ha) quedado sin efectos?")
SUPERADA_POR_RX = re.compile(
    r"(?:la|las)\s+(?:tesis|jurisprudencias?)(?:\s+de jurisprudencia|\s+aisladas?)?\s+(?P<lista>.{3,1500}?)"
    r"(?:se encuentran?|qued[óo]|quedaron|fue|fueron|ha sido|han sido|ha quedado|han quedado)\s+(?P<v>superad[ao]s?)\s+"
    r"(?:por|con)\s+(?:el criterio[^.]{0,160}?(?:que dio origen a|del que deriv[óo]|de la que deriv[óo])\s+)?(?:la|las)\s+"
    r"(?:tesis|jurisprudencias?)(?:\s+de jurisprudencia|\s+jurisprudencial(?:es)?)?\s+(?P<por>.{3,60})", re.I | re.S)
MODIF_A_ESTA_RX = re.compile(
    r"determin[óo] (?P<v>modificar|sustituir) la tesis\s+(?P<lista>.{3,40}?),.{0,300}?para quedar en los t[ée]rminos de "
    r"(?:esta|la presente) tesis", re.I)
MODIF_RESOLUCION_RX = re.compile(
    r"(?:solicitud de (?:modificaci[óo]n|sustituci[óo]n) de jurisprudencia|expediente varios)[^.]{0,250}?"
    r"determin[óo] (?P<v>modificar|abandonar|sustituir|interrumpir|dejar sin efectos) (?:el|los) criterios? contenidos? en "
    r"(?:la|las) (?:tesis|jurisprudencias?)(?: de jurisprudencia)?\s+(?P<lista>.{3,160}?)(?:,\s*(?:de|cuyos)\s+rubros?|,\s*derivad|"
    r"\s+para sostener|:|;)", re.I)

# Rubro de la tesis nueva.
RUBRO_RX = re.compile(
    r"[\[(][^\[\]()]{0,200}?(?P<v>ABANDONO|INTERRUPCI[ÓO]N|SUSTITUCI[ÓO]N|MODIFICACI[ÓO]N)(?P<parc>\s+PARCIAL)?"
    r"(?:\s+Y\s+MODIFICACI[ÓO]N)?\s+(?:DE|DEL)\s+(?:LA\s+|LAS\s+|LOS\s+)?(?:CRITERIOS?|TESIS|JURISPRUDENCIAS?)\b(?P<resto>.*)$",
    re.S)
RUBRO_SUPERADA_RX = re.compile(r"[\[(]\s*(?:JURISPRUDENCIA|TESIS)\s+(?P<lista>.+?)\s+SUPERADAS?\s*[\])]\.?\s*$", re.S)

# Lo que NO es pérdida: va a ver_jurisprudencia.
CONTRADICCION_NO_RX = re.compile(
    r"declar[óo]\s+(?P<t>inexistente|sin materia|improcedente)\s+(?:la\s+)?(?:contradicci[óo]n|denuncia)", re.I)
CONTENDIO_RX = re.compile(
    r"(?:esta tesis|la presente tesis|el criterio contenido en esta tesis)\s+(?:contendi[óo]|fue objeto de la denuncia|particip[óo])",
    re.I)
SOLICITUD_NEGADA_RX = re.compile(
    r"declar[óo]\s+(?:improcedente|infundada|sin materia|desechada)[^.]{0,40}la solicitud de (?:sustituci[óo]n|modificaci[óo]n)",
    re.I)


def es_jurisprudencia_clave(c: str) -> bool:
    return bool(re.search(r"/\s*J\.|\bJ/", c))


# --------------------------------------------------------------------------- extracción
class Extractor:
    def __init__(self, acervo: Acervo):
        self.A = acervo
        self.rel: List[dict] = []
        self.ver: Dict[str, dict] = {}
        self.sin_resolver: List[dict] = []
        self.origen_prec = collections.Counter()
        self._origen_actual = "qdrant"

    # --- helpers
    def _es_propia(self, reg: str, c: str) -> bool:
        b, s = norm_clave(c)
        sb, ss = self.A.clave_de(reg)
        return b == sb and (s is None or ss is None or s == ss)

    def _anterior(self, a: str, b: str) -> Optional[bool]:
        """¿a es anterior (o igual) a b? por fecha de publicación o, si falta, por el número de registro."""
        fa, fb = fecha_pub(self.A.por_reg.get(a)), fecha_pub(self.A.por_reg.get(b))
        if fa and fb:
            return fa <= fb
        oa, ob = orden_registro(a), orden_registro(b)
        if oa is None or ob is None:
            return None
        return oa <= ob

    def _ref(self, texto_clave: Optional[str], reg_self: str, rubro: Optional[str],
             registro_nota: Optional[str], anterior_a: Optional[str] = None) -> dict:
        """Resuelve la otra tesis de la relación a registro.
        anterior_a: si se da, la tesis resuelta debe ser anterior a ese registro (la afectada
        siempre es anterior a la tesis nueva que la abandona)."""
        out = {"clave": texto_clave, "rubro": rubro, "registro": None, "via": None, "confianza": None,
               "en_acervo": None}
        if registro_nota and registro_nota != reg_self:
            out["registro"], out["via"] = registro_nota, "registro_en_nota"
            out["confianza"] = "alta"
            out["en_acervo"] = registro_nota in self.A.por_reg
            return out
        r_clave = None
        if texto_clave:
            r, conf = self.A.resolver_clave(texto_clave, excluir=reg_self, rubro_hint=rubro)
            if r and anterior_a and self._anterior(r, anterior_a) is False:
                # misma clave en otra época (p. ej. 'I.9o.C.10 C' de 1995 contra 'I.9o.C.10 C (10a.)')
                out["confianza"] = "cronologia"
                r = None
                r_clave = "cronologia"
            if r:
                ok = self.A.rubro_coincide(r, rubro)
                if ok is False:
                    # La clave existe en el acervo pero el rubro citado es otro: probablemente
                    # es una tesis de otra época con la misma clave. Intentar por rubro.
                    r2 = self.A.resolver_rubro(rubro, excluir=reg_self) if rubro else None
                    if r2 and anterior_a and self._anterior(r2, anterior_a) is False:
                        r2 = None
                    if r2:
                        out.update(registro=r2, via="rubro", confianza="media", en_acervo=True)
                        return out
                    out.update(via="clave_rubro_distinto", confianza="dudosa", en_acervo=None)
                    return out
                out.update(registro=r, via="clave", confianza=conf, en_acervo=True,
                           duplicados=list(self.A.ultimos_duplicados))
                return out
            if not r_clave:
                out["confianza"] = conf
        if rubro:
            r = self.A.resolver_rubro(rubro, excluir=reg_self)
            if r and anterior_a and self._anterior(r, anterior_a) is False:
                r = None
                out["confianza"] = "cronologia"
            if r:
                out.update(registro=r, via="rubro", confianza="media", en_acervo=True)
                return out
        if out["en_acervo"] is None:
            out["en_acervo"] = False if out["confianza"] == "no_en_acervo" else None
        return out

    def _add(self, **k):
        k.setdefault("precedentes_de", self._origen_actual)
        self.rel.append(k)

    # --- por tesis
    def procesar(self, t: dict):
        reg = str(t.get("registro"))
        rubro = t.get("rubro") or ""
        prec_txt, origen = precedentes_completos(t)
        self.origen_prec[origen] += 1
        self._origen_actual = origen
        pars = parrafos(prec_txt)
        prec_todo = "\n\n".join(pars)
        self._rubro_nuevo(reg, rubro, prec_todo)
        self._tesis_sustituida(reg, prec_todo)
        for p in pars:
            self._parrafo(reg, p, prec_todo)

    def _rubro_nuevo(self, reg: str, rubro: str, prec: str):
        cola = rubro[-700:]
        m = RUBRO_RX.search(cola)
        if m:
            seg = cola[m.start():]
            cl = [c for c in claves_en(seg) if not self._es_propia(reg, c)]
            rq = re.search(r"RUBRO:?\s*[\"“'‘]([^\"”'’]{15,600})", seg)
            est = estado_de(m.group("v"))
            parc = bool(m.group("parc"))
            if not cl and rq:
                self._relacion_nueva(reg, None, rq.group(1), est, parc, "rubro", seg.strip()[:NOTA_BRUTA], prec)
            if not cl and not rq:
                self.sin_resolver.append({"por_registro": reg, "motivo": "rubro_sin_clave",
                                          "nota": seg.strip()[:NOTA_BRUTA]})
            for c in cl:
                self._relacion_nueva(reg, c, None, est, parc, "rubro", seg.strip()[:NOTA_BRUTA], prec)
            return
        m = RUBRO_SUPERADA_RX.search(cola)
        if m:
            for c in claves_en(m.group("lista")):
                if not self._es_propia(reg, c):
                    self._relacion_nueva(reg, c, None, "superada", False, "rubro",
                                         cola[m.start():].strip()[:NOTA_BRUTA], prec)

    def _tesis_sustituida(self, reg: str, prec: str):
        m = TESIS_SUSTITUIDA_RX.search(prec)
        if not m:
            return
        cuerpo = m.group("cuerpo")
        corte = re.search(r"\n\s*\n(?=(?:El Tribunal|La (?:Primera|Segunda) Sala|Nota|Tesis de jurisprudencia|"
                          r"Esta tesis|El Pleno|Los Magistrados))", cuerpo)
        if corte:
            cuerpo = cuerpo[:corte.start()]
        # sólo antes del primer rubro citado, para no tomar claves citadas dentro de él
        cabeza = re.split(r"de\s+(?:rubros?|t[íi]tulo)", cuerpo, maxsplit=1)[0]
        rq = RUBRO_Q_RX.search(cuerpo)
        nota = prec[m.start():m.start() + NOTA_BRUTA].strip()
        pos = claves_pos(cabeza)
        hubo = False
        for k, (c, i, f) in enumerate(pos):
            fin = pos[k + 1][1] if k + 1 < len(pos) else len(cuerpo)
            rn = registro_tras(cuerpo, f, fin if k + 1 < len(pos) else None)
            if self._es_propia(reg, c) and not (rn and rn != reg):
                self.sin_resolver.append({"por_registro": reg, "motivo": "autorreferencia", "afectada_clave": c,
                                          "patron": "tesis_sustituida", "nota": nota})
                continue
            hubo = True
            self._relacion_nueva(reg, c, rq.group(1) if (rq and len(pos) == 1) else None, "sustituida", False,
                                 "tesis_sustituida", nota, prec, registro_nota=rn)
        if not pos:
            if rq:
                self._relacion_nueva(reg, None, rq.group(1), "sustituida", False, "tesis_sustituida", nota, prec,
                                     registro_nota=registro_tras(cuerpo, 0, len(cuerpo)))
            else:
                self.sin_resolver.append({"por_registro": reg, "motivo": "tesis_sustituida_sin_clave", "nota": nota})

    def _parrafo(self, reg: str, p: str, prec: str):
        # 0) lo que no es pérdida
        if SOLICITUD_NEGADA_RX.search(p):
            return
        mno = CONTRADICCION_NO_RX.search(p)
        mco = CONTENDIO_RX.search(p)
        if mno or (mco and not re.search(r"abandon|interrump|superad|sustitu", p[:mco.start()], re.I)):
            self._ver_jurisprudencia(reg, p, mno.group("t") if mno else "contendio")
            # "declaró inexistente / sin materia / improcedente" y "contendió" no son pérdida de vigencia,
            # aunque de pasada digan que la postura de algún colegiado "quedó superada".
            return

        # 1) cita a terceras (o a sí misma por clave)
        for m in SUPERADA_POR_RX.finditer(p):
            lista = m.group("lista")
            pos = claves_pos(lista)
            if not pos or pos[0][1] > 40:
                continue   # el sujeto de "superada" no es una tesis nombrada ahí mismo
            cabeza = re.split(r",?\s+de\s+(?:rubros?|t[íi]tulo)|,?\s+publicad|,?\s+con n[úu]mero", lista, maxsplit=1)[0]
            afect = claves_en(cabeza)
            por = claves_en(m.group("por"))
            if not afect or not por:
                continue
            por_c = por[0]
            for c in afect:
                self._relacion_general(reg, c, por_c, "superada", False, "superada_por",
                                       recorte(p, m.start()), p, prec)
        for rx in (SIN_EFECTOS_RX, SIN_EFECTOS_B_RX):
            for m in rx.finditer(p):
                afect = claves_en(m.group("lista"))
                if not afect:
                    rq = RUBRO_Q_RX.search(p, m.start())
                    rq = rq or re.search(r"rubros?:?\s*[\"“'‘´]([^\"”'’]{15,600})", p[m.start():])
                    if rq:
                        r_af = self.A.resolver_rubro(rq.group(1))
                        if r_af and r_af != reg:
                            self._add(afectado=r_af, afectado_via="rubro", afectado_confianza="media",
                                      afectado_clave_citada=None,
                                      por={"clave": None, "rubro": None, "registro": None, "via": None,
                                           "confianza": None, "en_acervo": None},
                                      estado="sin_efectos", parcial=False, fuente="cita_tercera", citada_en=reg,
                                      patron="sin_efectos_rubro", nota=recorte(p, m.start()), desde=None,
                                      desde_origen=None, por_otros=[], por_resolucion=self._resolucion(p))
                resol = re.search(r"(solicitud de (?:modificaci[óo]n|sustituci[óo]n) de jurisprudencia\s+[\d/A-Z\-]+|"
                                  r"reforma constitucional[^,.;]{0,80})", p, re.I)
                for c in afect:
                    self._relacion_general(reg, c, None, "sin_efectos", False, "sin_efectos",
                                           recorte(p, m.start()), p, prec,
                                           por_resolucion=resol.group(1) if resol else None)
        for m in MODIF_A_ESTA_RX.finditer(p):
            for c in claves_en(m.group("lista")):
                if not self._es_propia(reg, c):
                    self._relacion_nueva(reg, c, None, estado_de(m.group("v")), False, "modificada_a_esta",
                                         recorte(p, m.start()), prec)
        for m in MODIF_RESOLUCION_RX.finditer(p):
            afect = claves_en(m.group("lista"))
            if not afect:
                continue
            nueva_es_self = bool(re.search(r"(?:la presente|esta) tesis deriva", p, re.I)) or \
                bool(re.match(r"\s*Solicitud de (?:modificaci[óo]n|sustituci[óo]n) de jurisprudencia", prec, re.I))
            for c in afect:
                if self._es_propia(reg, c):
                    # la afectada es ella misma: el reemplazo es la tesis de rubro citado después
                    rq = RUBRO_Q_RX.search(p, m.end())
                    self._relacion_propia(reg, None, rq.group(1) if rq else None, estado_de(m.group("v")), False,
                                          "modificacion_de_si", recorte(p, m.start()), p)
                elif nueva_es_self:
                    self._relacion_nueva(reg, c, None, estado_de(m.group("v")), False, "modificacion",
                                         recorte(p, m.start()), prec)
                else:
                    self._relacion_general(reg, c, None, estado_de(m.group("v")), False, "modificacion_tercera",
                                           recorte(p, m.start()), p, prec)

        # 2) la tesis habla de sí misma
        for nombre, rx in PROPIA:
            m = rx.search(p)
            if not m:
                continue
            # "…quien formuló voto concurrente en el que se aparta del criterio contenido en la presente tesis":
            # es un voto, no una pérdida de vigencia.
            if re.search(r"\bvotos?\b(?:\s+(?:concurrente|particular|aclaratorio|razonado|de minor[íi]a))?[^.;]{0,60}$",
                         p[max(0, m.start() - 90):m.start()], re.I) and not re.search(r"al resolver", p[max(0, m.start() - 90):m.start()], re.I):
                self.sin_resolver.append({"por_registro": None, "citada_en": reg, "motivo": "voto_no_perdida",
                                          "patron": nombre, "nota": recorte(p, m.start())})
                break
            # nota copiada de otra tesis: "…se aparta del criterio sostenido en esta tesis publicada en …,
            # registro digital: 171750" dentro de un registro que no es el 171750.
            ajena = re.search(r"(?:esta|la presente) tesis,?\s+publicada en[^\"“]{0,260}?registro digital:?\s*(\d{6,7})",
                              p[m.start():])
            if ajena and ajena.group(1) != reg:
                self.sin_resolver.append({"por_registro": None, "citada_en": reg, "afectada_registro": ajena.group(1),
                                          "motivo": "nota_de_otra_tesis", "patron": nombre,
                                          "nota": recorte(p, m.start())})
                break
            # "La fracción IV del artículo 115 … a que se hace referencia en esta tesis, fue modificada mediante
            # decreto": cambió la norma interpretada, no el criterio.
            if re.search(r"(?:referencia|refiere|alcances se fijaron|interpretaci[óo]n|interpretad[oa]s?|citad[oa]s?|"
                         r"mencionad[oa]s?|analizad[oa]s?)\s+(?:en|de)\s*$", p[max(0, m.start() - 60):m.start()], re.I):
                self.sin_resolver.append({"por_registro": None, "citada_en": reg, "motivo": "norma_reformada_no_perdida",
                                          "patron": nombre, "nota": recorte(p, m.start())})
                break
            v = m.groupdict().get("v") or ("dejo" if nombre == "dejo_de_ser_obligatoria" else "")
            est = estado_de(v) if nombre != "dejo_de_ser_obligatoria" else self._estado_por_contexto(p)
            if nombre == "cancelada":
                est = "sin_efectos"
            parc = bool((m.groupdict().get("parc") or "").strip() or (m.groupdict().get("parc2") or "").strip())
            cola = p[m.end():]
            cl = [c for c in claves_en(cola) if not self._es_propia(reg, c)]
            rq = RUBRO_Q_RX.search(cola)
            if re.search(r"^\s*para que (?:guardara|guarde|refleje|reflejara|coincida|coincidiera)", cola, re.I):
                est = "texto_sustituido"   # corrección del texto publicado, no cambio de criterio
            if (nombre == "cancelada" or est == "sin_efectos") and re.search(r"republicad|como consta", cola, re.I):
                # "…como consta en la tesis republicada…": la republicada es el aviso de cancelación, no un reemplazo
                cl, rq, cola = [], None, ""
            self._relacion_propia(reg, cl[0] if cl else None, rq.group(1) if rq else None, est, parc, nombre,
                                  recorte(p, m.start()), p, otros=cl[1:4], cola=cola)
            break

        # 3) la tesis declara lo que reemplaza
        for nombre, rx in NUEVA:
            for m in rx.finditer(p):
                v = m.group("v")
                if nombre == "ya_no_obligatoria":
                    est = self._estado_por_contexto(p + " " + prec[:600])
                else:
                    est = estado_de(v)
                parc = bool((m.groupdict().get("parc") or "").strip())
                cola = p[m.end():]
                mismo = nombre in ("texto_se_sustituye", "rubro_y_texto_sustituyen", "tesis_sustituye_a_la")
                if mismo:
                    est = "texto_sustituido"
                if nombre == "texto_se_sustituye":
                    pos = [(m.group("c"), m.start("c") - m.end(), m.end("c") - m.end())]
                else:
                    cabeza = re.split(r"de\s+(?:rubros?|t[íi]tulos?)|,?\s+publicad|,?\s+que aparece|,?\s+cuyos?\s+rubros?|"
                                      r",?\s+con (?:el )?rubro|,?\s+del propio|,?\s+sustentad|,?\s+derivad", cola,
                                      maxsplit=1)[0]
                    pos = claves_pos(cabeza) or claves_pos(cola[:300])[:1]
                nota = recorte(p, m.start())
                rq = RUBRO_Q_RX.search(p, m.end()) if len(pos) <= 1 else None
                if not pos and rq:
                    self._relacion_nueva(reg, None, rq.group(1), est, parc, nombre, nota, prec)
                elif not pos:
                    self.sin_resolver.append({"por_registro": reg, "motivo": f"{nombre}_sin_clave", "nota": nota})
                for k, (c, i, f) in enumerate(pos):
                    fin = pos[k + 1][1] if k + 1 < len(pos) else None
                    rn = registro_tras(cola, f, fin)
                    if not mismo and self._es_propia(reg, c) and not (rn and rn != reg):
                        self.sin_resolver.append({"por_registro": reg, "motivo": "autorreferencia", "afectada_clave": c,
                                                  "patron": nombre, "nota": nota})
                        continue
                    self._relacion_nueva(reg, c, rq.group(1) if rq else None, est, parc, nombre, nota, prec,
                                         mismo_numero=mismo, registro_nota=rn)

    def _estado_por_contexto(self, texto: str) -> str:
        t = plegar(texto).lower()
        for rx, e in [(r"sustituci|sustitu", "sustituida"), (r"interrump|interrupc", "interrumpida"),
                      (r"abandon", "abandonada"), (r"modificaci", "modificada"), (r"superad", "superada")]:
            if re.search(rx, t):
                return e
        return "superada"

    # --- constructores de relación
    def _relacion_propia(self, reg, clave, rubro, estado, parcial, patron, nota, parrafo, otros=(), cola=None):
        regs = [x for x in REGISTRO_RX.findall(cola if cola is not None else parrafo) if x != reg]
        ref = self._ref(clave, reg, rubro, regs[0] if regs else None)
        desde, origen = self._desde_propia(parrafo, ref.get("registro"), cola, clave)
        self._add(afectado=reg, afectado_via="propia", por=ref, estado=estado, parcial=parcial,
                  fuente="nota_propia", patron=patron, nota=nota, desde=desde, desde_origen=origen,
                  fecha_resolucion=self._fecha_resolucion(parrafo),
                  por_otros=[{"clave": c} for c in otros],
                  por_resolucion=self._resolucion(parrafo))

    def _relacion_nueva(self, reg, clave_afectada, rubro_afectada, estado, parcial, patron, nota, prec,
                        mismo_numero=False, registro_nota=None):
        if mismo_numero and clave_afectada:
            r, conf = self.A.resolver_clave(clave_afectada, excluir=reg)
            ref = {"clave": clave_afectada, "rubro": None, "registro": r, "via": "clave", "confianza": conf,
                   "en_acervo": bool(r)}
        else:
            ref = self._ref(clave_afectada, reg, rubro_afectada, registro_nota, anterior_a=reg)
        if ref.get("registro") and ref["registro"] not in self.A.por_reg:
            self.sin_resolver.append({"por_registro": reg, "afectada_clave": clave_afectada,
                                      "afectada_registro": ref["registro"], "estado": estado,
                                      "motivo": "afectada_fuera_del_acervo", "patron": patron, "nota": nota})
            return
        if not ref.get("registro"):
            self.sin_resolver.append({"por_registro": reg, "afectada_clave": clave_afectada,
                                      "afectada_rubro": (rubro_afectada or "")[:200], "estado": estado,
                                      "motivo": ref.get("confianza") or "sin_clave", "patron": patron, "nota": nota})
            return
        desde, origen = self._desde_nueva(reg, prec)
        yo = {"clave": self.A.por_reg[reg].get("clave_tesis"), "rubro": self.A.por_reg[reg].get("rubro"),
              "registro": reg, "via": "propia", "confianza": "alta", "en_acervo": True}
        for i, af in enumerate([ref["registro"]] + [d for d in ref.get("duplicados", []) if d != reg]):
            self._add(afectado=af, afectado_via=ref["via"] if i == 0 else "duplicado_misma_clave_y_rubro",
                      afectado_confianza=ref["confianza"], afectado_clave_citada=clave_afectada, por=yo, estado=estado,
                      parcial=parcial, fuente="tesis_nueva", patron=patron, nota=nota, desde=desde,
                      desde_origen=origen, por_otros=[], por_resolucion=None)

    def _relacion_general(self, reg, clave_afectada, clave_por, estado, parcial, patron, nota, parrafo, prec,
                          por_resolucion=None):
        # ¿es ella misma la afectada o la nueva?
        if self._es_propia(reg, clave_afectada):
            self._relacion_propia(reg, clave_por, None, estado, parcial, patron, nota, parrafo)
            return
        if clave_por and self._es_propia(reg, clave_por):
            self._relacion_nueva(reg, clave_afectada, None, estado, parcial, patron, nota, prec)
            return
        ra = self._ref(clave_afectada, reg, None, None)
        if not ra.get("registro"):
            self.sin_resolver.append({"por_registro": None, "citada_en": reg, "afectada_clave": clave_afectada,
                                      "estado": estado, "motivo": ra.get("confianza") or "sin_clave",
                                      "patron": patron, "nota": nota})
            return
        rp = self._ref(clave_por, ra["registro"], None, None) if clave_por else \
            {"clave": None, "rubro": None, "registro": None, "via": None, "confianza": None, "en_acervo": None}
        desde, origen = (fecha_pub(self.A.por_reg.get(rp.get("registro") or "")), "publicacion_reemplazo") \
            if rp.get("registro") else self._fecha_en(parrafo)
        for i, af in enumerate([ra["registro"]] + ra.get("duplicados", [])):
            self._add(afectado=af, afectado_via=ra["via"] if i == 0 else "duplicado_misma_clave_y_rubro",
                      afectado_confianza=ra["confianza"], afectado_clave_citada=clave_afectada, por=rp,
                      estado=estado, parcial=parcial, fuente="cita_tercera", citada_en=reg, patron=patron, nota=nota,
                      desde=desde, desde_origen=origen, por_otros=[],
                      por_resolucion=por_resolucion or self._resolucion(parrafo))

    def _fecha_resolucion(self, p: str) -> Optional[str]:
        """Fecha de la resolución que le quitó la vigencia, si la nota la da."""
        ses = re.search(r"en sesiones de ([^.]{10,200}?),?\s+respectivamente", p, re.I)
        if ses:
            fs = [iso(*f) for f in FECHA_RX.findall(ses.group(1) + " de " + (re.findall(r"\d{4}", ses.group(1)) or ["0"])[-1])
                  if f[1].lower() in MESES]
            if fs:
                return max(fs)
        m = re.search(r"(?:en sesi[óo]n (?:de|del)|por ejecutoria (?:de fecha |del? )?|resuelta[^,]{0,120}? el|"
                      r"al resolver[^.]{0,160}?,?\s*el)\s*(\d{1,2})\s+de\s+(\w+)\s+de\s+(\d{4})", p, re.I)
        if m and m.group(2).lower() in MESES:
            return iso(*m.groups())
        return fecha_palabras(p)

    def _resolucion(self, p: str) -> Optional[str]:
        m = re.search(r"(contradicci[óo]n de (?:tesis|criterios)\s+[\d/]+(?:-[A-Z]{2})?|solicitud de (?:sustituci[óo]n|"
                      r"modificaci[óo]n) de jurisprudencia\s+[\d/]+(?:-[A-Z]{2})?|amparo (?:directo|en revisi[óo]n|"
                      r"directo en revisi[óo]n)\s+[\d/]+|revisi[óo]n en incidente de suspensi[óo]n\s+[\d/]+|"
                      r"conflicto competencial\s+[\d/]+|expediente varios\s+[\d/]+(?:-[A-Z]{2})?|"
                      r"acci[óo]n de inconstitucionalidad\s+[\d/]+|controversia constitucional\s+[\d/]+|"
                      r"(?:recurso de )?queja\s+[\d/]+|recurso de reclamaci[óo]n\s+[\d/]+|amparo directo\s+[\d/]+)", p, re.I)
        return m.group(1) if m else None

    def _fecha_en(self, p: str) -> Tuple[Optional[str], Optional[str]]:
        m = re.search(r"a partir del?\s+(?:lunes|martes|mi[ée]rcoles|jueves|viernes|s[áa]bado|domingo)?\s*"
                      r"(\d{1,2})\s+de\s+(\w+)\s+de\s+(\d{4})", p, re.I)
        if m and m.group(2).lower() in MESES:
            return iso(*m.groups()), "nota_a_partir_de"
        m = re.search(r"publicad[ao]s?\s+(?:el\s+)?(?:en el Semanario Judicial de la Federaci[óo]n\s+)?"
                      r"(?:del?\s+)?(?:lunes|martes|mi[ée]rcoles|jueves|viernes)\s+(\d{1,2})\s+de\s+(\w+)\s+de\s+(\d{4})",
                      p, re.I)
        if m and m.group(2).lower() in MESES:
            return iso(*m.groups()), "nota_publicacion_reemplazo"
        m = MES_ANIO_RX.search(p)
        if m:
            return f"{m.group(2)}-{MESES[m.group(1).lower()]:02d}", "nota_mes_publicacion"
        m = re.search(r"(?:en sesi[óo]n (?:de|del)|por ejecutoria (?:de fecha |del? )?)\s*(\d{1,2})\s+de\s+(\w+)\s+de\s+(\d{4})",
                      p, re.I)
        if m and m.group(2).lower() in MESES:
            return iso(*m.groups()), "nota_fecha_resolucion"
        return None, None

    def _desde_propia(self, p: str, por_reg: Optional[str], cola: Optional[str] = None,
                      clave: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
        m = re.search(r"a partir del?\s+(?:lunes|martes|mi[ée]rcoles|jueves|viernes)?\s*(\d{1,2})\s+de\s+(\w+)\s+de\s+(\d{4})",
                      p, re.I)
        if m and m.group(2).lower() in MESES:
            return iso(*m.groups()), "nota_a_partir_de"
        if por_reg and fecha_pub(self.A.por_reg.get(por_reg)):
            return fecha_pub(self.A.por_reg.get(por_reg)), "publicacion_reemplazo"
        # publicación del reemplazo, sólo en el texto que sigue a su clave (no la de la propia tesis)
        tras = None
        if cola and clave and clave in cola:
            tras = cola[cola.index(clave):]
        if tras:
            m = re.search(r"(?:lunes|martes|mi[ée]rcoles|jueves|viernes)\s+(\d{1,2})\s+de\s+(\w+)\s+de\s+(\d{4})", tras, re.I)
            if m and m.group(2).lower() in MESES:
                return iso(*m.groups()), "nota_publicacion_reemplazo"
            m = MES_ANIO_RX.search(tras)
            if m:
                return f"{m.group(2)}-{MESES[m.group(1).lower()]:02d}", "nota_mes_publicacion_reemplazo"
        # fecha de la resolución que le quitó la vigencia
        m = re.search(r"(?:en sesi[óo]n (?:de|del)|por ejecutoria (?:de fecha |del? )?|,\s*el|al resolver[^,]{0,80},?\s*el)\s*"
                      r"(\d{1,2})\s+de\s+(\w+)\s+de\s+(\d{4})", p, re.I)
        if m and m.group(2).lower() in MESES:
            return iso(*m.groups()), "nota_fecha_resolucion"
        f = fecha_palabras(p)
        if f:
            return f, "nota_fecha_resolucion"
        return None, None

    def _desde_nueva(self, reg: str, prec: str) -> Tuple[Optional[str], Optional[str]]:
        m = re.search(r"se considera de aplicaci[óo]n obligatoria a partir del?\s+(?:lunes|martes|mi[ée]rcoles|jueves|viernes)?"
                      r"\s*(\d{1,2})\s+de\s+(\w+)\s+de\s+(\d{4})", prec, re.I)
        if m and m.group(2).lower() in MESES:
            return iso(*m.groups()), "nota_a_partir_de"
        f = fecha_pub(self.A.por_reg.get(reg))
        if f:
            return f, "publicacion_tesis_nueva"
        return None, None

    # --- ver_jurisprudencia
    def _ver_jurisprudencia(self, reg: str, p: str, tipo: str):
        cl = [c for c in claves_en(p) if es_jurisprudencia_clave(c) and not self._es_propia(reg, c)]
        if not cl:
            return
        tipo = {"inexistente": "inexistente", "sin materia": "sin_materia", "improcedente": "improcedente"}.get(
            plegar(tipo).lower(), "contendio")
        refs = []
        for c in cl[:4]:
            r, conf = self.A.resolver_clave(c, excluir=reg)
            refs.append({"clave": c, "registro": r, "confianza": conf})
        d = self.ver.setdefault(reg, {"clave": self.A.por_reg[reg].get("clave_tesis"), "entradas": []})
        d["entradas"].append({"tipo": tipo, "jurisprudencias": refs, "nota": p[:400],
                              "menciona_abandono": bool(re.search(r"abandon|superad|interrump", p, re.I))})


# --------------------------------------------------------------------------- consolidación
PRIORIDAD_FUENTE = {"nota_propia": 0, "tesis_nueva": 1, "cita_tercera": 2}
PRIORIDAD_CONF = {"alta": 0, "media": 1, None: 2, "ambigua": 3, "dudosa": 3, "no_en_acervo": 3}


def direccion_ok(A: Acervo, afectado: str, por: Optional[str]) -> Optional[bool]:
    if not por or por not in A.por_reg or afectado not in A.por_reg:
        oa, op = orden_registro(afectado), orden_registro(por) if por else None
        if oa is None or op is None:
            return None
        return op > oa
    fa, fp = fecha_pub(A.por_reg[afectado]), fecha_pub(A.por_reg[por])
    if fa and fp:
        return fp >= fa
    oa, op = orden_registro(afectado), orden_registro(por)
    return None if oa is None or op is None else op > oa


def consolidar(A: Acervo, rel: List[dict]) -> Tuple[Dict[str, dict], List[dict]]:
    por_af: Dict[str, List[dict]] = collections.defaultdict(list)
    descartes = []
    for r in rel:
        if not r["afectado"] or r["afectado"] == (r["por"] or {}).get("registro"):
            descartes.append({**r, "motivo_descarte": "afectado_igual_reemplazo"})
            continue
        por_af[r["afectado"]].append(r)
    out: Dict[str, dict] = {}
    for af, rs in por_af.items():
        rs.sort(key=lambda r: (PRIORIDAD_FUENTE[r["fuente"]], 0 if (r["por"] or {}).get("registro") else 1,
                               PRIORIDAD_CONF.get(r.get("afectado_confianza", "alta"), 2)))
        p = rs[0]
        por = p["por"] or {}
        por_reg = por.get("registro")
        # si la fuente principal no resolvió el reemplazo, pedirlo prestado a otra fuente que sí
        if not por_reg:
            for r in rs[1:]:
                if (r["por"] or {}).get("registro"):
                    por = r["por"]
                    por_reg = por.get("registro")
                    break
        t_af = A.por_reg.get(af, {})
        aviso_en = None
        if por_reg and por_reg in A.por_reg and p["estado"] != "texto_sustituido" and \
                norm_clave(A.por_reg[por_reg].get("clave_tesis") or "")[0] == norm_clave(t_af.get("clave_tesis") or "")[0] and \
                A.rubro_coincide(por_reg, t_af.get("rubro")):
            # "…según se desprende de la que con el número VI.2o.C.591 C aparece publicada en mayo de 2019": es la
            # misma tesis republicada con el aviso, no un criterio nuevo.
            aviso_en, por_reg, por = por_reg, None, {}
        t_por = A.por_reg.get(por_reg or "", {})
        fuentes = sorted({r["fuente"] for r in rs}, key=lambda f: PRIORIDAD_FUENTE[f])
        otros = []
        for r in rs:
            rr = (r["por"] or {}).get("registro")
            if rr and rr != por_reg and rr not in otros:
                otros.append(rr)
        out[af] = {
            "estado": p["estado"],
            "parcial": any(r["parcial"] for r in rs if r["estado"] == p["estado"]) or p["parcial"],
            "por_registro": por_reg,
            "por_clave": t_por.get("clave_tesis") or por.get("clave"),
            "por_rubro": (t_por.get("rubro") or por.get("rubro") or None),
            "por_en_acervo": bool(t_por) if por_reg else None,
            "por_resolucion": p.get("por_resolucion"),
            "desde": p.get("desde"),
            "desde_origen": p.get("desde_origen"),
            "fuente": p["fuente"],
            "fuentes": fuentes,
            "nota": (p.get("nota") or "")[:NOTA_BRUTA],
            "clave": t_af.get("clave_tesis"),
            "tipo": t_af.get("tipo"),
            "epoca": t_af.get("epoca"),
            "patron": p["patron"],
            "confianza_resolucion": "alta" if p["fuente"] == "nota_propia" and por.get("confianza") in ("alta", None)
            else (p.get("afectado_confianza") if p["fuente"] != "nota_propia" else por.get("confianza")),
            # True: el reemplazo es posterior (lo normal). False: el reemplazo es ANTERIOR — p. ej. un
            # colegiado abandona su tesis para alinearse con una jurisprudencia ya existente. No es error.
            "reemplazo_posterior": direccion_ok(A, af, por_reg),
            "fecha_resolucion": p.get("fecha_resolucion"),
            "otros_reemplazos": otros[:5],
            "precedentes_de": p.get("precedentes_de"),
            "aviso_republicado_en": aviso_en,
        }
        if p["fuente"] == "cita_tercera":
            out[af]["citada_en"] = p.get("citada_en")
        # Una jurisprudencia nueva obliga desde el lunes siguiente a su publicación; si su propia nota lo dice,
        # ésa es la fecha en que la anterior dejó de obligar.
        if por_reg in A.por_reg and out[af]["desde_origen"] == "publicacion_reemplazo":
            txt, _ = precedentes_completos(A.por_reg[por_reg])
            mo = re.search(r"se considera de aplicaci[óo]n obligatoria a partir del?\s+(?:lunes|martes|mi[ée]rcoles|jueves|"
                           r"viernes)?\s*(\d{1,2})\s+de\s+(\w+)\s+de\s+(\d{4})", txt, re.I)
            if mo and mo.group(2).lower() in MESES:
                out[af]["desde"], out[af]["desde_origen"] = iso(*mo.groups()), "obligatoria_desde_segun_reemplazo"
        # Si el reemplazo es anterior (el colegiado se alineó con una jurisprudencia ya existente), la
        # vigencia se pierde cuando se resolvió el asunto que lo declara, no cuando se publicó el reemplazo.
        fr = p.get("fecha_resolucion")
        if fr and (out[af]["reemplazo_posterior"] is False or not out[af]["desde"] or
                   (out[af]["desde_origen"] in ("publicacion_reemplazo", "nota_mes_publicacion_reemplazo",
                                                "nota_publicacion_reemplazo") and fr > (out[af]["desde"] or "")
                    and p["patron"] in ("abandono_el_criterio_de_esta", "criterio_fue", "tesis_fue"))):
            out[af]["desde"], out[af]["desde_origen"] = fr, "nota_fecha_resolucion"
    # Segunda pasada: si el "reemplazo" resultó ser la misma tesis republicada con el aviso, la fecha útil es la
    # de la resolución que el aviso relata (en la propia nota o en la del registro republicado).
    for af, v in out.items():
        av = v.get("aviso_republicado_en")
        if not av:
            continue
        fr = v.get("fecha_resolucion") or (out.get(av) or {}).get("fecha_resolucion")
        if fr:
            v["desde"], v["desde_origen"] = fr, "nota_fecha_resolucion"
        else:
            v["desde"], v["desde_origen"] = fecha_pub(A.por_reg.get(av)), "publicacion_del_aviso"
    return out, descartes


# --------------------------------------------------------------------------- el archivo compacto
def cortar(texto: Optional[str], tope: int) -> Optional[str]:
    """Literal hasta `tope` caracteres, cortado en palabra entera y con «…» si
    se cortó. El prototipo cortaba a pelo y la nota de 2009817 acababa en «pu»:
    un modelo que lee «pu» completa lo que le parece."""
    if not texto:
        return texto or None
    t = re.sub(r"\s+", " ", texto).strip()
    if len(t) <= tope:
        return t
    corte = t[:tope - 1]
    esp = corte.rfind(" ")
    if esp > tope * 0.6:
        corte = corte[:esp]
    return corte.rstrip(" ,;:") + "…"


def compacto(idx: Dict[str, dict]) -> Dict[str, dict]:
    """Del índice completo, sólo lo que lee `vigencia_tesis.py`. El resto
    (patrón, confianza, fuentes, otros reemplazos…) se queda en --informes."""
    out = {}
    for reg, v in sorted(idx.items(), key=lambda kv: int(kv[0])):
        out[reg] = {
            "estado": v["estado"],
            "parcial": bool(v["parcial"]),
            "por_registro": v.get("por_registro"),
            "por_clave": v.get("por_clave"),
            "por_rubro": cortar(v.get("por_rubro"), TOPE_RUBRO),
            "por_resolucion": v.get("por_resolucion"),
            "desde": v.get("desde"),
            "fuente": v["fuente"],
            "nota": cortar(v.get("nota"), TOPE_NOTA),
        }
    return out


def main():
    global SJF_CACHE
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--env", default=str(RAIZ / ".env"),
                    help="archivo .env con QDRANT_URL y QDRANT_API_KEY (sólo se leen)")
    ap.add_argument("--sjf-cache", default=SJF_CACHE_OMISION,
                    help="carpeta con {registro}.json del SJF para los `precedentes` cortados")
    ap.add_argument("--salida", default=str(SALIDA))
    ap.add_argument("--tesis-cache", default=None,
                    help="volcado jsonl de la v3: se reutiliza si existe, si no se escribe")
    ap.add_argument("--informes", default=None,
                    help="carpeta para el índice completo, ver_jurisprudencia, sin_resolver y conteos")
    a = ap.parse_args()

    SJF_CACHE = a.sjf_cache
    if not os.path.isdir(SJF_CACHE):
        print(f"⚠️ No existe la caché del SJF ({SJF_CACHE}): las 1,370 tesis con `precedentes` cortado "
              "se leerán cortadas y pueden faltar notas.", file=sys.stderr)

    tesis = cargar(Path(a.env), Path(a.tesis_cache) if a.tesis_cache else None)
    A = Acervo(tesis)
    E = Extractor(A)
    for t in tesis:
        E.procesar(t)
    idx, descartes = consolidar(A, E.rel)

    # ver_jurisprudencia sólo para las que no perdieron vigencia de forma expresa
    ver = {r: v for r, v in E.ver.items() if r not in idx}
    sin, vistos = [], set()
    for s in E.sin_resolver:
        k = (s.get("por_registro"), s.get("citada_en"), s.get("afectada_clave"), s.get("motivo"),
             s.get("nota", "")[:80])
        if k in vistos:
            continue
        vistos.add(k)
        sin.append(s)

    tesis_c = compacto(idx)
    salida = {
        "generado": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "fuente": (f"{COLECCION} (Qdrant, sólo lectura: rubro y precedentes) + SJF para los "
                   f"{sum(1 for t in tesis if len(t.get('precedentes') or '') >= TOPE_QDRANT)} "
                   f"`precedentes` cortados a {TOPE_QDRANT} car.; scripts/vigencia_tesis_generar.py"),
        "n": len(tesis_c),
        "tesis": tesis_c,
    }
    ruta = Path(a.salida)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(salida, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    cnt = collections.Counter(v["estado"] for v in idx.values())
    meta = {
        "coleccion": COLECCION,
        "tesis_recorridas": len(tesis),
        "afectadas": len(idx),
        "por_estado": dict(cnt.most_common()),
        "por_fuente": dict(collections.Counter(v["fuente"] for v in idx.values()).most_common()),
        "con_reemplazo_resuelto": sum(1 for v in idx.values() if v["por_registro"]),
        "sin_reemplazo_resuelto": sum(1 for v in idx.values() if not v["por_registro"]),
        "con_dos_o_mas_fuentes": sum(1 for v in idx.values() if len(v["fuentes"]) > 1),
        "parciales": sum(1 for v in idx.values() if v["parcial"]),
        "descartes": len(descartes),
        "afectadas_nombradas_sin_resolver": len(sin),
        "precedentes_origen": dict(E.origen_prec),
        "afectadas_con_nota_tomada_del_sjf": sum(1 for v in idx.values() if v.get("precedentes_de") == "sjf"),
        "ver_jurisprudencia_fuera_del_indice": len(ver),
        "bytes_salida": ruta.stat().st_size,
    }
    if a.informes:
        d = Path(a.informes)
        d.mkdir(parents=True, exist_ok=True)

        def j(nombre, datos):
            (d / nombre).write_text(json.dumps(datos, ensure_ascii=False, indent=1), encoding="utf-8")
        j("vigencia_completo.json", dict(sorted(idx.items(), key=lambda kv: int(kv[0]))))
        j("vigencia_ver_jurisprudencia.json", ver)
        j("vigencia_sin_resolver.json", sin)
        j("vigencia_meta.json", meta)
    print(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"✓ {len(tesis_c)} tesis con pérdida de vigencia → {ruta}")


if __name__ == "__main__":
    main()
