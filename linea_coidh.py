"""
La Corte IDH en el chat: la línea jurisprudencial y los casos que cita el abogado — 25-sep-2026.

    import linea_coidh
    linea_coidh.pregunta_por_linea("¿Dónde nace el control de convencionalidad?")
    → ("control_convencionalidad", ["origen"])
    linea_coidh.pregunta_por_linea("prisión preventiva oficiosa para extorsión")
    → ("control_convencionalidad", ["tema:prision_preventiva_oficiosa"])

POR QUÉ EXISTE
--------------
Iurexia citaba el ¶124 de Almonacid con la etiqueta de Trabajadores Cesados:
el texto venía de un cuadernillo con las etiquetas corridas un caso (1,408 de
1,453 trozos comprobables, plan §2). Este módulo es la mitad del chat del plan
de ingesta (reingesta/coidh/plan_ingesta_coidh.md §3.5-3.7); la otra mitad,
qué caso nombra un texto, vive en `coidh_catalogo.py`. Tiene la forma de
`doctrina.py`: reglas de entrada medidas, bloque para el modelo y un fallo que
nunca cuesta la consulta.

Tres cosas, ninguna con embeddings ni modelos:
  · la LÍNEA: `pregunta_por_linea` decide si la pregunta es de la figura
    (alias exactos con límite de palabra: «pena convencional» no dispara) o de
    un tema mexicano (prisión preventiva oficiosa, arraigo, fuero militar…) y
    `traer_linea` trae por `llave` los hitos de `datos/lineas_coidh.json`;
  · los CASOS CITADOS: `traer_citados` trae el párrafo que pidió el abogado
    —y un vecino a cada lado, por `orden`—; sin párrafo, los resolutivos y los
    tres párrafos más citados; sin ingesta, la ficha del catálogo (cita y URL
    oficial, sin texto inventado);
  · los BLOQUES para el modelo, que se AÑADEN después de
    `format_results_as_xml` (revisión A.3 del plan): `reorder_by_hierarchy`
    ordena por (nivel, −puntaje) y ponía el párrafo oficial detrás de toda la
    Constitución y de todas las leyes, y al cuadernillo (nivel 0) delante de
    la sentencia. Aquí manda la cronología.

LA PUERTA (revisión A.1 y B.10, medida el 25-sep-2026)
-----------------------------------------------------
Sobre 4,000 preguntas reales, la ficha abriría algo en 11 (0.28 %: 5 por
«convencionalidad», 5 por prisión preventiva oficiosa, 1 por fuero militar) y
el resolvedor en 1 (0.03 %, recortada a 4,000 caracteres). `is_ddhh_query` se
dispara en el 12.8 %: por eso la puerta NO lo usa. COIDH_ACTIVO la gobierna:
«off», «admins» (por omisión, el piloto) u «on».

LA COLECCIÓN
------------
Alias `coidh` (física `coidh_parrafos_p1`), payload de la sección 3.4 del plan
tal como lo escribe `scripts/ingesta_coidh.py`. Existe desde el 25-sep-2026
(16,825 puntos de 58 resoluciones). Si deja de responder, todo esto devuelve
vacío, lo dice en el registro y la consulta sigue exactamente igual que antes.

EL PILAR (25-sep-2026, noche)
-----------------------------
David preguntó si los Tribunales Colegiados pueden hacer control difuso sobre
las normas del juicio de origen, y el chat le dio como vigentes la P. IX/2015
y la P. X/2015, ABANDONADAS desde feb-2022 por la P./J. 2/2022 (reg. 2024159).
La nota de abandono estaba en nuestro Qdrant; ninguna ruta la ponía delante
del modelo, y la sustituta salía en el puesto 38 por similitud. Aquí se
arregla la mitad que toca a la línea (la otra mitad, la vigencia de CUALQUIER
tesis recuperada, es del otro frente, en main.py):
  · la ficha trae ahora una `cronologia` de 200 entradas verificadas (sólo
    «verificado» o «parcial»), seis `tramos` con su resumen, los `cortes` de
    cada fuente y la `recepcion_mx` con vigencia, fuerza y reemplazo;
  · `seleccionar` prioriza por relevancia a la pregunta, vigencia y fuerza —ya
    no por el orden del archivo, que dejaba fuera la P./J. 20/2014 y la
    P./J. 2/2022 en la pregunta de David— y si entra una tesis abandonada
    entra su sustituta, y al revés;
  · la detección abre para las preguntas reales que no abrían (control difuso
    de los Colegiados, amparo contra reformas constitucionales, cláusulas
    pétreas, inconvencionalidad de la Constitución) y, para lo que las palabras
    no alcanzan, `sondear` pregunta a la colección `lineas` (hoy no existe:
    devuelve None);
  · el selector de fuentes con sólo «jurisprudencia» ya no cierra la línea: la
    abre en modo «solo_mx», sin nada de la Corte IDH, la CIDH, la ONU, la
    doctrina ni las reformas.
"""
from __future__ import annotations

import hashlib
import html
import json
import os
import re
import time
import unicodedata
import uuid
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

COLECCION = "coidh"          # el alias; la física es coidh_parrafos_p1 (ingesta_coidh.py)
SILO = "coidh"               # = COLECCION: _parse_results y /cita usan el nombre consultado
RUTA_LINEAS = Path(__file__).resolve().parent / "datos" / "lineas_coidh.json"
RUTA_COPIAS = Path(__file__).resolve().parent / "datos" / "coidh_copias.json"

# El resolvedor tarda p99 23 ms con 30 mil caracteres, pero con escritos
# pegados se dispara por el escrito y no por la pregunta (revisión B.8): sobre
# la PREGUNTA, y recortada, como se midió el 0.03 %.
TOPE_PREGUNTA = 4000
MAX_CASOS = 6                # resultados del resolvedor que se atienden
MAX_LLAVES_POR_CASO = 5      # «párrs. 123 a 160» no trae 38 párrafos
MAX_UNIDADES_CITADAS = 24    # techo de lo que entra por el resolvedor
MAX_PARTES = 6               # ventanas de un mismo párrafo partido (~500 tokens cada una)
MAX_HITOS = 16               # ~240 tokens por hito (cl100k): la línea entera, 3.5-6.5 mil
MAX_TESIS = 10
# Con México en la pregunta entran también tesis y la cronología mexicana: el
# tope de hitos baja para que la línea entera quede por debajo de ~8 mil
# tokens (medido con cl100k en test_linea_pilar.py, 25-sep-2026).
MAX_HITOS_MEXICO = 8
MAX_TESIS_LINEA = 8          # la línea entera con México: las tesis del núcleo y las que más pesan
MAX_CRONO = 5                # entradas de <cronologia> (~120 tokens cada una)
MAX_PAREJAS = 3              # sustitutas o abandonadas que entran por encima de MAX_TESIS
# Cuántos hitos se le garantizan a cada fase pedida que no cubren ya el núcleo
# (el origen) ni los hitos fijos del estado actual con México. Revisión
# adversarial, 26-sep-2026: con México y «actual» el tope dejaba a lo de cupo
# sin lugar y la línea saltaba de Cesados (24-nov-2006) a Tzompaxtle
# (7-nov-2022), mientras la instrucción seguía pidiendo «→ ejes».
MIN_POR_FASE = 2
# Radilla Pacheco ¶339-341 (23-nov-2009): la sentencia que trajo el control a
# México (el Varios 912/2010 la cumple). Entra siempre que la pregunta sea de
# México y pida el origen, la evolución o el concepto. Revisión adversarial,
# 26-sep-2026: el filtro de «reitera» sacaba el ¶339 y los ¶340-341 perdían
# contra cualquier «amplía» de otro país; entraba Boyce (Barbados) y no Radilla.
RADILLA = ("C-209|s|339", "C-209|s|340", "C-209|s|341")
# El aporte (y la nota de vigencia) de una entrada: hasta 55 palabras cortando
# en un fin de frase, no a media. Revisión adversarial, 26-sep-2026: el recorte
# a 32 palabras dejaba la AI 130/2019 en «…la interpretación conforme del art.
# 19 no tuvo…» y se perdía «mayoría y la P./J. 20/2014 quedó intacta». 55 y no
# 45: es el largo con el que armar_pilar.py escribió los aportes («las primeras
# oraciones, hasta ~55 palabras»); con 45 el voto de Franco en la CT 293/2011
# perdía justo lo que propone. Hoy todos entran enteros; el corte por frase es
# para los que vengan más largos.
MAX_PALABRAS_APORTE = 55
# EL PRESUPUESTO DEL BLOQUE, ahora aplicado y no sólo medido. Revisión
# adversarial, 26-sep-2026: la pregunta nueva de David («Traza la línea… y dime
# qué posturas serias hay para inaplicar restricciones constitucionales como la
# prisión preventiva oficiosa») abría línea, estado actual y dos temas y medía
# 10,285 tokens; nada lo acotaba. traer_linea quita piezas en el orden de
# sel["recortables"] (lo que entró por puntuación y lo que los temas traen de
# más; ver seleccionar) y nunca el núcleo, el estado actual, Radilla, lo
# garantizado a cada fase ni las parejas de un abandono. Sin tiktoken en
# producción se mide en caracteres: cl100k da 2.74-2.91 caracteres por token en
# estos bloques (2.87-2.91 en los grandes, que son los que se recortan); con
# 2.8 se peca de largo.
#
# RE-MEDIDO el 26-sep-2026 (cl100k, test_linea_pilar.py): con Radilla ¶339-341
# y dos hitos de evolución, la línea entera con México pasa de ~7.9 a ~9.1 mil
# tokens, y la de dos temas se queda en ~9.5 mil sin perder el núcleo, el estado
# actual ni las posturas que pide. El tope de la línea sube a 10 mil —hoy
# aplicado; antes 8 mil sólo medidos y 10.3 mil sin tope—. El de un tema solo
# (sin fase), de 5.5 a 6.5 mil: la prisión preventiva oficiosa trae ahora la
# CC 3/2026 y sus aportes y notas de vigencia enteros (~5.9 mil); con 5.5 o 6
# perdía el extracto de la supervisión que deja el punto abierto.
#
# VARIOS TEMAS SIN FASE (reverificación, 26-sep-2026, hallazgo 14.1): cada
# tema pedido trae ahora sus hitos esenciales protegidos (_esenciales_tema),
# su primera vuelta de tesis y de cronología, su advertencia y su tensión:
# ~1,200-1,700 tokens propios. Con 6,500 para todo, «prisión preventiva
# oficiosa y restricciones constitucionales expresas en el parámetro de
# regularidad» (tres temas; 10,008 sin recorte) conservaba los hitos pero
# perdía el AG 2/2024, la IX.P. J/2 P, el art. 166 LA y la CC 3/2026: el
# estado mexicano de la prisión preventiva. Por eso 1,500 más por cada tema
# después del primero, sin pasar de PRESUPUESTO_LINEA: dos temas, 8,000; tres,
# 9,500; cuatro o más, 10,000. Un tema solo sigue en 6,500. Medido con Qdrant
# real (cl100k): dos temas 7,326-7,671; tres o cuatro, 8,833-9,052.
# SUBE A 12,500 (26-sep-2026, al abrir la línea a todos). Con 10 mil, «Traza la
# línea… y dime qué posturas serias hay para inaplicar restricciones
# constitucionales» perdía la orden de Tzompaxtle sobre la prisión preventiva
# oficiosa (resolutivo 8) y, con 11 mil, todavía la doctrina de Cano López y
# Rodríguez Manzo, el proyecto de Pardo (recepción 3/2023), el voto de Franco y
# la CC 3/2026: justo las posturas que pide. Con 12,500 entran (11,926 tokens
# medidos con Qdrant real); sólo queda fuera el voto de Cossío y la doctrina de
# Silva García. Las demás preguntas de David no cambian (≤ 9,060). Sólo paga
# este tope quien pide una fase de la línea: 0 de 336 consultas reales la abren.
PRESUPUESTO_LINEA = 12500
PRESUPUESTO_TEMA = 6500
PRESUPUESTO_POR_TEMA = 1500
CAR_POR_TOKEN = 2.8

# La colección de la línea para la sonda semántica (scripts/lineas_qdrant.py).
# Alias; la física es `lineas_p1` (creada el 26-sep-2026: 208 puntos).
LINEAS = "lineas"
# Coseno de text-embedding-3-small. Prefiere no abrir a abrir en falso (una
# línea que no toca cuesta ~6-9 mil tokens en la respuesta). Se mueve con
# LINEAS_UMBRAL sin desplegar.
# CALIBRADO el 26-sep-2026 con la colección creada (208 puntos): de 12 preguntas
# que deben abrir la línea y no casan por palabras, 10 pasan de 0.60 (0.605-
# 0.764; se quedan «evolución de la interpretación conforme…» 0.592 y «¿una
# autoridad administrativa puede inaplicar…?» 0.533); de 60 consultas reales
# ajenas, ninguna (máx. 0.592). scratchpad calibrar_lineas.py / calibracion_lineas.json.
UMBRAL_LINEAS = 0.60

# La precisión histórica que el plan corrigió (§ «Sobre el origen»): la
# expresión es de los votos de García Ramírez desde 2003; la Corte en pleno la
# formula en Almonacid ¶124 (26-sep-2006) y la hace ex officio en Cesados ¶128
# (24-nov-2006). Estos tres entran SIEMPRE que entra la línea.
NUCLEO = ("C-101|v:garcia-ramirez|27", "C-154|s|124", "C-158|s|128")

_SEG_LLAVE = {"s": "sentencia", "r": "resolutivos", "c": "considerandos"}


# ═══════════════════════════════════════════════════════════════ interruptor

def modo() -> str:
    """COIDH_ACTIVO: «off» | «admins» (por omisión) | «on».

    Un valor que no se entiende cae a «admins», nunca a «on»: equivocarse
    hacia arriba abre el piloto a todos."""
    v = (os.getenv("COIDH_ACTIVO") or "admins").strip().lower()
    if v in ("off", "false", "0", "no", "apagado"):
        return "off"
    if v in ("on", "true", "1", "si", "sí", "todos"):
        return "on"
    return "admins"


def alcance_por_fuentes(fuentes: Optional[Iterable[str]] = None) -> Optional[str]:
    """Qué parte de la línea deja pasar el selector de fuentes del abogado.

      · «completa» — el rubro «constitucional» está encendido (o no hay
        selector): la Corte IDH, la CIDH, la ONU, la doctrina y las reformas;
      · «solo_mx» — «constitucional» apagado pero «jurisprudencia» encendida:
        sólo tesis y resoluciones mexicanas (SCJN, Plenos Regionales, TCC);
      · None — los dos apagados: la línea no entra.

    POR QUÉ (25-sep-2026): la pregunta de David sobre los Colegiados llevaba
    sólo «jurisprudencia» y la puerta cerraba la línea entera, con la P./J.
    2/2022 dentro. La Corte IDH sí es del rubro «constitucional» (así lo dice
    `fuentes_elegidas.instruccion()`), pero las tesis de la SCJN son
    jurisprudencia nacional: cerrarlas por la Corte IDH era tirar lo mexicano
    con lo interamericano."""
    try:
        import fuentes_elegidas as fs
    except Exception:
        return "completa"
    elegidas = frozenset(fuentes) if fuentes is not None else None
    if not fs.excluye("constitucional", elegidas):
        return "completa"
    if not fs.excluye("jurisprudencia", elegidas):
        return "solo_mx"
    return None


# ═══════════════════════════════════════════════════════════════ utilidades

def _plegar(s: str) -> str:
    """Minúsculas y sin acentos (los disparadores de la ficha están escritos así)."""
    d = unicodedata.normalize("NFD", s or "")
    return "".join(c for c in d if unicodedata.category(c) != "Mn").lower()


def _norm(s: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", _plegar(s)))


def _esc(v: Any) -> str:
    return html.escape("" if v is None else str(v), quote=True)


_MESES = ("enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
          "septiembre", "octubre", "noviembre", "diciembre")


def fecha_larga(iso: Optional[str]) -> str:
    """«2006-09-26» → «26 de septiembre de 2006» (así la escribe la Corte)."""
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", str(iso or ""))
    if not m:
        return str(iso or "")
    return f"{int(m.group(3))} de {_MESES[int(m.group(2)) - 1]} de {m.group(1)}"


def ficha_id(clave: str) -> str:
    """Id estable de una ficha sin punto en Qdrant (catálogo o hito no ingerido).

    Mismo patrón que los ids de la ingesta (uuid de un md5): así [Doc ID] y el
    sello funcionan también para lo que todavía no se ingirió, y /cita lo puede
    reconstruir sin guardar nada (`ficha_por_id`)."""
    return str(uuid.UUID(hashlib.md5(f"coidh-ficha|{clave}".encode("utf-8")).hexdigest()))


# ═══════════════════════════════════════════════════════════════ el PDF del visor

# LA COPIA QUE SÍ SE DEJA LEER (25-sep-2026). El visor decía «No se pudo abrir
# el PDF aquí» en todas las sentencias: Cloudflare de corteidh.or.cr reta con
# 403 a TODO cliente automático, también al proxy de Vercel. Por eso se
# subieron copias verificadas —el MISMO sha1 que el PDF con el que la ingesta
# mapeó las páginas; cotejado para las 58 resoluciones de la colección— al
# bucket público `legal-docs/CorteIDH/`, donde ya viven la Constitución y los
# tratados. El manifiesto `datos/coidh_copias.json` dice cuál es la copia de
# cada URL oficial: {url_oficial: {copia, sha1, bytes, doc_id}}.
#
# El contrato: `pdf_url` = la copia (lo que abre el visor), `url_oficial` = la
# de corteidh.or.cr (la cita y el enlace «ver en el sitio de la Corte»), y
# `pdf_sha1` = el de la copia, que es el que versiona la caché del visor. Sin
# copia en el manifiesto, `pdf_url` sigue siendo la URL oficial, como antes.

_HOSTS_CORTEIDH = ("www.corteidh.or.cr", "corteidh.or.cr")


def canon_corteidh(url: Optional[str]) -> Optional[str]:
    """`https://www.corteidh.or.cr/<ruta>`, sin «#page» ni consulta; None si no
    es de la Corte. La misma forma que `canonCorteIDH` del frontend
    (src/lib/proxyPdf.ts): http→https, con «www.», la ruta tal cual."""
    if not url:
        return None
    from urllib.parse import urlsplit
    try:
        s = urlsplit(str(url).strip().split("#")[0])
    except ValueError:
        return None
    if s.scheme not in ("http", "https") or (s.hostname or "") not in _HOSTS_CORTEIDH or not s.path:
        return None
    return f"https://www.corteidh.or.cr{s.path}"


@lru_cache(maxsize=1)
def copias() -> Dict[str, Dict[str, Any]]:
    """datos/coidh_copias.json por URL oficial canónica, una vez por proceso.

    Perezoso y tolerante, como `lineas()`: si falta o está roto, el visor vuelve
    a pedir la URL oficial (la de antes) y la consulta sigue igual."""
    try:
        crudo = json.loads(RUTA_COPIAS.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"   ⚖️ COIDH: no pude leer {RUTA_COPIAS.name} ({type(e).__name__}); el visor pedirá la URL oficial")
        return {}
    salida: Dict[str, Dict[str, Any]] = {}
    for url, d in (crudo or {}).items():
        k = canon_corteidh(url)
        copia = (d or {}).get("copia")
        # Sólo una dirección https entera: un manifiesto a medio escribir no
        # puede mandar al visor a una ruta relativa.
        if k and isinstance(copia, str) and copia.startswith("https://"):
            salida[k] = dict(d)
    return salida


def _copia_por_sha1(sha1: str) -> Optional[Dict[str, Any]]:
    """La copia cuyo sha1 es `sha1` (58 entradas: recorrerlas cuesta nada y
    así no hay una segunda caché que se desfase de `copias()`)."""
    s = str(sha1 or "").strip().lower()
    return next((d for d in copias().values() if s and str(d.get("sha1") or "").lower() == s), None)


def pdf_de(url_oficial: Optional[str], sha1: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """(pdf_url, pdf_sha1) para el visor a partir de la URL oficial.

    Con copia: la copia y SU sha1. Sin copia: la URL oficial sin «#page» y el
    sha1 que ya se tuviera (el del payload de la ingesta).

    Se busca por la URL oficial y, si no está, por el sha1 del PDF que se
    troceó: mismo sha1 es el MISMO archivo, byte por byte. Hace falta porque
    cuatro resoluciones (C-217, C-218, C-221, C-253) se trocearon con el «_esp»
    cuando el catálogo pide «_esp1/_esp2» (ingesta --aceptar-alterno): un
    payload escrito antes de ese cambio trae la URL del catálogo. Por número
    de caso NO se busca: una resolución puede tener más de un PDF, y abrir
    otro distinto del citado es la peor falla del visor (ver resolver_pdf).

    Y si el payload trae sha1 y la copia de esa URL tiene OTRO (revisión del
    25-sep-2026): la Corte reemplazó el PDF después de trocearlo, o la copia
    se subió de otra descarga. Las páginas y el ancla se midieron en el
    archivo del payload, no en ése; manda el sha1, y sin copia con ese sha1 se
    queda la URL oficial, como antes. Hoy las 58 coinciden (test_coidh_chat)."""
    oficial = (str(url_oficial).split("#")[0] or None) if url_oficial else None
    c = copias().get(canon_corteidh(oficial) or "") if oficial else None
    s = str(sha1 or "").strip().lower()
    if c and s and str(c.get("sha1") or "").strip().lower() not in ("", s):
        c = None
    if not c and s:
        c = _copia_por_sha1(s)
    if c:
        return c["copia"], (c.get("sha1") or sha1)
    return oficial, sha1


# ═══════════════════════════════════════════════════════════════ la ficha

@lru_cache(maxsize=1)
def lineas() -> Dict[str, Any]:
    """datos/lineas_coidh.json, una vez por proceso. Si falta o está roto, la
    capa se apaga sola (sin figuras) y la consulta sigue igual."""
    try:
        return json.loads(RUTA_LINEAS.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"   ⚖️ COIDH: no pude leer {RUTA_LINEAS.name} ({type(e).__name__}); sin línea")
        return {"figuras": {}}


def figuras() -> Dict[str, Any]:
    return lineas().get("figuras") or {}


@lru_cache(maxsize=8)
def _alias_rx(fid: str) -> Tuple[re.Pattern, ...]:
    f = figuras().get(fid) or {}
    rxs = []
    for a in f.get("alias") or []:
        cuerpo = r"\s+".join(re.escape(t) for t in _plegar(a).split())
        if cuerpo:
            rxs.append(re.compile(r"\b" + cuerpo + r"\b"))
    for a in f.get("alias_rx") or []:
        rxs.append(re.compile(a))
    # «inconvencional(idad)» pasó de `alias_amplios` a `alias_rx` el
    # 25-sep-2026, con límite de palabra: la pregunta de David sobre la
    # «inconvencionalidad» de la Constitución no abría. Medido en
    # `prueba_disparadores` de la ficha (test_linea_pilar.py).
    return tuple(rxs)


@lru_cache(maxsize=32)
def _tema_rx(fid: str, tid: str) -> Tuple[Tuple[re.Pattern, ...], Tuple[re.Pattern, ...]]:
    t = ((figuras().get(fid) or {}).get("temas_mx") or {}).get(tid) or {}
    return (tuple(re.compile(x) for x in t.get("disparadores") or []),
            tuple(re.compile(x) for x in t.get("excluir") or []))


# Qué quiere saber de la figura. Sobre texto plegado (sin acentos).
_RX_FASES = {
    "origen": re.compile(
        r"\b(origen(es)?|nace|nacio|nacimiento|surge|surgio|surgimiento|primera\s+vez|acun\w*"
        r"|antecedentes?|desde\s+cuando|cuando\s+(aparece|aparecio|nace|nacio|surge|surgio|empez\w+)"
        r"|quien\s+(acuno|creo|invento|lo\s+creo|uso\s+primero)|historia|genesis|precursor\w*)\b"),
    "evolucion": re.compile(
        r"\b(evolucion\w*|ha\s+cambiado|cambio|cambios|desarrollo|linea\s+jurisprudencial|trayectoria"
        r"|ejes|alcances?|ampli\w+|ex\s+officio)\b"),
    "actual": re.compile(
        r"\b(hoy|actual\w*|vigente|estado\s+actual|reciente\w*|ultim[oa]s?|ahora|todavia|aun|sigue"
        r"|postura\w*|20(2[3-9]|3\d))\b"),
    "concepto": re.compile(
        r"\b(que\s+es|concepto|definicion|define|en\s+que\s+consiste|naturaleza|significa|explica\w*"
        r"|como\s+funciona)\b"),
}
# Los ejes de la línea: una pregunta por un eje trae TODOS sus hitos —la lista
# de quiénes están obligados ES la respuesta a «¿quiénes están obligados?»
# (pregunta de oro 5), reiteraciones incluidas—. La clave es la `fase` del
# hito en la ficha.
_RX_EJES = {
    "sujetos": re.compile(
        r"\b(obligad\w*|quien(es)?\s+(debe\w*|esta\w*|ejerce\w*|lo\s+ejerce\w*|puede\w*)|autoridad(es)?"
        r"|organos?|sujetos|legislador\w*|ministerio\s+publico)\b"),
    "parámetro": re.compile(
        r"\b(parametro|opinion(es)?\s+consultiva\w*|otros\s+tratados|bloque\s+de\s+convencionalidad"
        r"|corpus\s+juris)\b"),
    "efectos": re.compile(
        r"\b(efectos|res\s+interpretata|res\s+judicata|cosa\s+(juzgada|interpretada)"
        r"|norma\s+convencional\s+interpretada|vinculant\w*)\b"),
    "forma": re.compile(r"\b(modelo|difuso|concentrado)\b"),
    "complementariedad": re.compile(r"\b(complementari\w*|subsidiari\w*)\b"),
    "cumplimiento": re.compile(r"\b(cumpl\w*|sin\s+reforma|reforma\s+legal)\b"),
}
# México en la pregunta: trae su resumen, sus artículos constitucionales y más
# recepción. No abre la puerta por sí solo. Desde el 25-sep-2026 también los
# tribunales mexicanos por su nombre: la pregunta de David («¿los tribunales
# colegiados pueden…?») no decía «SCJN» y la capa de México no se encendía.
_RX_MEXICO = re.compile(
    r"\b(mexic\w*|scjn|suprema\s+corte|varios\s+912|ct\s*293|contradiccion\s+de\s+tesis\s+293"
    r"|tribunal(es)?\s+colegiados?|colegiados|juicio\s+de\s+origen|regularidad\s+constitucional"
    r"|plenos?\s+regional(es)?|primera\s+sala|segunda\s+sala|juez(es)?\s+de\s+distrito)\b")


def pregunta_por_linea(texto: str) -> Optional[Tuple[str, List[str]]]:
    """(figura, fases) si la pregunta es de una línea curada; None si no.

    `fases` trae una o varias de «origen», «evolucion», «actual», «concepto»
    cuando casó un alias de la figura (por omisión «concepto»); además
    «eje:<fase>» si pregunta por un eje (sujetos, parámetro, efectos…),
    «mexico» si nombra a México o a la SCJN, y una entrada «tema:<id>» por
    cada tema mexicano de `temas_mx` que disparó (revisión B.10): «prisión
    preventiva oficiosa», «arraigo» penal, «fuero militar»… no dicen
    «convencionalidad» y son justo donde la línea importa. Un tema solo trae
    sus 2-4 hitos, no la línea entera.

    Límite de palabra en todo: «pena convencional», «interés convencional» o
    «convención colectiva» no disparan (`no_disparan` de la ficha); el arraigo
    mercantil o civil tampoco (`excluir` del tema).

    Si sólo casó un tema y la pregunta dice además qué fase quiere («¿cuál es
    la postura ACTUAL de la SCJN sobre el control difuso?»), esas fases se
    pasan con la marca «tema_primero»: el tema manda y la fase ordena, sin
    arrastrar el núcleo del origen interamericano (25-sep-2026)."""
    if not texto or not texto.strip():
        return None
    p = _plegar(texto[:TOPE_PREGUNTA])
    for fid, f in figuras().items():
        casa_figura = any(rx.search(p) for rx in _alias_rx(fid))
        temas = []
        donde: Dict[str, int] = {}
        for tid in (f.get("temas_mx") or {}):
            dispara, excluye = _tema_rx(fid, tid)
            pos = [m.start() for m in (rx.search(p) for rx in dispara) if m]
            if pos and not any(rx.search(p) for rx in excluye):
                temas.append(tid)
                donde[tid] = min(pos)
        # En el orden en que la pregunta los nombra, no en el del archivo: con
        # dos temas, el reparto por turnos empieza por el primero que se nombra
        # («…inaplicar restricciones constitucionales como la prisión
        # preventiva oficiosa»: restricciones primero; 26-sep-2026).
        temas.sort(key=lambda t: donde[t])
        if not casa_figura and not temas:
            continue
        fases: List[str] = []
        mx = ["mexico"] if _RX_MEXICO.search(p) else []
        if casa_figura:
            ejes = [f"eje:{k}" for k, rx in _RX_EJES.items() if rx.search(p)]
            fases = [k for k, rx in _RX_FASES.items() if rx.search(p)] or (["evolucion"] if ejes else ["concepto"])
            # «Desde el nacimiento hasta la postura actual» es la línea entera:
            # pide también lo de en medio (26-sep-2026: «Traza la línea
            # cronológica desde el nacimiento… hasta la postura actual…» no
            # traía ningún hito de la Corte IDH entre 2006 y 2022).
            if "origen" in fases and "actual" in fases and "evolucion" not in fases:
                fases.insert(fases.index("actual"), "evolucion")
            fases += ejes + mx
        else:
            pedidas = [k for k, rx in _RX_FASES.items() if rx.search(p) and k != "concepto"]
            fases = pedidas + mx + (["tema_primero"] if pedidas else [])
        return fid, fases + [f"tema:{t}" for t in temas]
    return None


# ── Qué hitos entran según lo que se pregunta ─────────────────────────────
_FASES_HITO = {
    "origen": {"antecedente", "formulación en pleno", "precisión"},
    "evolucion": {"formulación en pleno", "precisión", "contenido", "contenido (méxico)", "sujetos",
                  "sistematización", "parámetro", "complementariedad", "efectos", "forma", "cumplimiento"},
    "actual": {"estado actual", "estado actual (méxico)"},
    "concepto": {"formulación en pleno", "precisión", "sistematización", "forma", "parámetro", "efectos"},
}
# Las fases que son EJES de la línea (las de _RX_EJES: quién está obligado,
# parámetro, efectos, forma…): si ninguno entró, la instrucción no los pide
# (26-sep-2026).
_FASES_EJE = frozenset(_RX_EJES)
# Lo que aporta primero: la arista dice si el hito origina, amplía o sólo reitera.
_PESO_ARISTA = {"origina": 0, "amplía": 1, "precisa": 2, "tensiona": 3, "antecede": 4, "reitera": 5, "recibe": 6}
# Artículos de la CPEUM que acompañan la línea cuando se pregunta por México
# (los de la tensión: prisión preventiva, restricciones, arraigo).
_CONSTITUCION_MEXICO = ("art. 19 párr. 2", "art. 105", "art. 107")


def _hitos_por_llave(fid: str) -> Dict[str, Dict[str, Any]]:
    f = figuras().get(fid) or {}
    out = {h["llave"]: h for h in f.get("hitos") or []}
    for t in (f.get("temas_mx") or {}).values():
        for x in t.get("extra") or []:
            out.setdefault(x["llave"], {**x, "fase": x.get("fase") or "tema", "arista": x.get("arista") or "precisa",
                                        "doc_id": x["llave"].split("|")[0], "aporte": x.get("aporte") or "",
                                        "verificado": x.get("verificado", False)})
    return out


# ── La cronología del pilar (25-sep-2026) ─────────────────────────────────
# Qué tramos de la narración (ver «tramos» en la ficha) tocan cada fase o
# tema: suben de prioridad las entradas de esos tramos, y un tema con fase
# («tema_primero») trae el resumen de su tramo.
_TRAMOS_FASE = {"origen": ("t1",), "concepto": ("t1", "t2"), "evolucion": ("t2", "t3", "t4"),
                "actual": ("t6",), "mexico": ("t3", "t4", "t6")}
_TRAMO_TEMA = {"restricciones_constitucionales": "t5", "amparo_reformas_constitucionales": "t5",
               "prision_preventiva_oficiosa": "t6", "arraigo": "t6", "control_ex_officio_mx": "t4",
               "control_difuso_jueces_locales": "t3"}
# Modo «solo_mx» (el abogado apagó «constitucional»): de la cronología sólo
# pasan tesis y resoluciones mexicanas. Ni la Corte IDH, ni la CIDH, ni la
# ONU, ni votos, proyectos, doctrina o reformas.
# «pleno_regional_resolucion» (26-sep-2026): la CC 3/2026 del Pleno Regional
# Centro-Norte, resuelta pero sin tesis publicada (la SCJN pidió suspender su
# publicación).
_TIPOS_SOLO_MX = frozenset({"scjn_tesis", "scjn_resolucion", "acuerdo_general", "pleno_regional_tesis",
                            "pleno_regional_resolucion", "tcc_tesis"})
_TIPOS_CORTE_IDH = ("corte_idh_sentencia", "corte_idh_voto", "corte_idh_oc", "corte_idh_supervision")
_PESO_VIGENCIA = {"vigente": 2.0, "pendiente": 1.0, "superada_en_parte": 0.5, "modificada": 0.0, "no_aplica": 0.0,
                  "historica": -0.5, "superada": -1.0, "abandonada": -1.0}
_PESO_FUERZA_TESIS = {"jurisprudencia_obligatoria": 2.0, "plenos_regionales": 1.0, "tesis_aislada": 0.5, "tcc": 0.0}
_PESO_FUERZA = {"obligatoria": 1.5, "norma": 1.5, "orientadora": 0.5, "recomendacion": 0.5, "historica": 0.0,
                "voto": 0.0, "doctrina": 0.0, "proyecto": 0.0, "no_publicada": 0.0}
# Cuántas de cada clase entran por puntuación (lo que pide un tema entra
# aparte): que una pregunta general no se llene de recomendaciones o votos.
_LIMITE_CLASE = {"recomendacion": 2, "scjn_voto": 2, "doctrina": 2, "scjn_proyecto": 2}
_STOP = frozenset(
    "de la el los las del en que se por con para una uno unos unas sus como esta este estos estas esas esos hay "
    "ser son fue han sin entre cual cuales donde cuando sobre desde hasta tiene tienen puede pueden debe deben "
    "mas muy tambien pero porque segun ante bajo cada otra otro otras otros dicho dicha todo toda todos todas "
    "cuya cuyo asi solo "
    # Las palabras de FASE dicen qué parte de la línea se quiere, no de qué
    # trata: «¿dónde nace…?» subía la 2a./J. 163/2017 porque su aporte dice
    # «el matiz nace en el ADR 583/2015» (medido el 25-sep-2026).
    "nace nacio surge surgio origen origenes actual actualmente vigente vigentes evolucion linea postura "
    "historia hoy ahora".split())


def cronologia(fid: str) -> List[Dict[str, Any]]:
    """La cronología verificada de la figura (vacía si la ficha no la trae)."""
    return (figuras().get(fid) or {}).get("cronologia") or []


@lru_cache(maxsize=8)
def _crono_por_id(fid: str) -> Dict[str, Dict[str, Any]]:
    return {e["id"]: e for e in cronologia(fid)}


def entrada_resuelta(fid: str, e: Dict[str, Any]) -> Dict[str, Any]:
    """Una entrada de la cronología con los datos de su hito si sólo es una
    referencia (`ref_hito`): URL, página, párrafo, extracto y aporte viven en
    el hito, no se duplican en la cronología."""
    ll = e.get("ref_hito")
    if not ll:
        return e
    h = _hitos_por_llave(fid).get(ll) or {}
    out = dict(e)
    for k in ("url_oficial", "pagina", "parrafo", "extracto", "aporte"):
        if out.get(k) in (None, "") and h.get(k) not in (None, ""):
            out[k] = str(h[k]) if k in ("pagina", "parrafo") else h[k]
    if not h.get("verificado", True) and out.get("verificacion", {}).get("estado") == "verificado":
        out["verificacion"] = dict(out["verificacion"], estado="parcial")
    return out


@lru_cache(maxsize=4096)
def _raices(texto: str) -> frozenset:
    """Raíces de 6 letras de las palabras con contenido (sin acentos): lo justo
    para que «colegiados» case con «Colegiados» y «reclamado» con «reclamados»."""
    return frozenset(w[:6] for w in re.findall(r"[a-z0-9]+", _plegar(texto or "")) if len(w) >= 4 and w not in _STOP)


def _texto_entrada(e: Dict[str, Any]) -> str:
    return " ".join(str(e.get(k) or "") for k in ("titulo", "aporte", "extracto", "clave", "organo"))


def _texto_tesis(r: Dict[str, Any], e: Optional[Dict[str, Any]]) -> str:
    return " ".join(str(x or "") for x in (r.get("rubro_abreviado"), r.get("porque"), r.get("clave"),
                                           (e or {}).get("titulo"), (e or {}).get("aporte")))


@lru_cache(maxsize=4)
def _idf(fid: str) -> Dict[str, float]:
    """Peso de cada raíz en el corpus de la línea (cronología + recepción): una
    raíz que está en todas («contro», «consti») no distingue nada; «colegi»,
    «difuso» o «reclam» sí. log(N / df), calculado una vez por proceso."""
    import math
    textos = [_texto_entrada(entrada_resuelta(fid, e)) for e in cronologia(fid)]
    textos += [_texto_tesis(r, None) for r in (figuras().get(fid) or {}).get("recepcion_mx") or []]
    df: Counter = Counter()
    for t in textos:
        df.update(_raices(t))
    n = max(1, len(textos))
    return {r: math.log(n / c) for r, c in df.items()}


def _relevancia(fid: str, raices_q: frozenset, texto: str) -> float:
    if not raices_q:
        return 0.0
    idf = _idf(fid)
    return sum(idf.get(r, 0.0) for r in raices_q & _raices(texto))


def _por_turnos(listas: Sequence[Sequence[str]]) -> List[Tuple[str, int, int]]:
    """Una de cada lista por vuelta, sin repetir: [(id, vuelta, lista)].

    POR QUÉ (revisión adversarial, 26-sep-2026): lo que piden los temas y el
    estado actual se CONCATENABA y se cortaba después. Con dos temas, el
    primero en el orden del archivo (prisión preventiva) se quedaba el cupo y el
    de restricciones perdía su doctrina y sus votos —justo las «posturas
    serias» que pedía David—, y el resultado dependía del orden de las claves
    del JSON, no de la pregunta."""
    colas = [list(x) for x in listas]
    vistos: set = set()
    out: List[Tuple[str, int, int]] = []
    vuelta = 0
    while any(colas):
        for k, c in enumerate(colas):
            while c:
                x = c.pop(0)
                if x not in vistos:
                    vistos.add(x)
                    out.append((x, vuelta, k))
                    break
        vuelta += 1
    return out


# Resolutivo → el párrafo de la misma sentencia que dice lo mismo (y más).
# Sólo pares cotejados en el texto de coidh; un resolutivo sin par aquí es
# esencial aunque haya otro párrafo que ordene (reverificación, 26-sep-2026).
_RESOLUTIVO_REPITE = {"C-482|r|14": "C-482|s|301"}


def _esenciales_tema(fid: str, llaves: Sequence[str], dentro: Iterable[str] = ()) -> List[str]:
    """Los hitos de un tema que el presupuesto nunca quita: los que ORDENAN
    (postura «ordena_adecuar» en la cronología: García Rodríguez ¶300 y ¶301,
    Tzompaxtle ¶118, los resolutivos de adecuar o suprimir) y los que fijan
    una tensión (están en `a_favor` o `en_contra` de alguna: García Rodríguez
    ¶303, Tzompaxtle ¶219). Si el tema no tiene ninguno así (control ex
    officio, fuero militar), el primero de su lista: cada tema pedido conserva
    al menos un hito de la Corte IDH.

    Un RESOLUTIVO que ordena deja de ser esencial sólo si en `dentro` (lo fijo
    de la selección) está el párrafo que DICE LO MISMO, según la tabla
    verificada _RESOLUTIVO_REPITE: García Rodríguez ¶301 dice lo del resolutivo
    14 y más («incluyendo sus disposiciones constitucionales»). Sigue
    entrando; sólo se puede recortar, y después de lo de segunda vuelta.
    Antes se suponía que CUALQUIER párrafo que ordena de la misma sentencia
    repetía al resolutivo, y así Tzompaxtle ¶118 (el principio del art. 27 de
    la Convención de Viena) dejaba sin proteger al resolutivo 8, que es la
    orden de adecuar la prisión preventiva oficiosa (¶¶212-213, 217-219): en
    «Traza la línea…» no entraba la orden de Tzompaxtle sobre la PPO
    (reverificación, 26-sep-2026). Cabe con el tope de 12,500.

    POR QUÉ (reverificación del 26-sep-2026, hallazgo 14.1): el recorte
    dejaba en cero los hitos interamericanos de las preguntas de dos o más
    temas sin fase; ver el orden de recortables en seleccionar."""
    crono = _crono_por_id(fid)
    en_tension = {ll for t in (figuras().get(fid) or {}).get("tensiones") or []
                  for ll in (t.get("a_favor") or []) + (t.get("en_contra") or [])}

    def ordena(ll: str) -> bool:
        return (crono.get(ll) or {}).get("postura") == "ordena_adecuar"

    def seg(ll: str) -> str:
        return ll.split("|")[1] if ll.count("|") >= 2 else ""
    dentro = set(dentro)
    salida = [ll for ll in llaves
              if ll in en_tension or (ordena(ll) and not (seg(ll) == "r" and _RESOLUTIVO_REPITE.get(ll) in dentro))]
    return salida or list(llaves[:1])


def _elegir_registros(fid: str, q: frozenset, reales: Sequence[str], temas: Sequence[str], mexico: bool,
                      tema_primero: bool, ids: Sequence[str], solo_mx: bool) -> Dict[str, Any]:
    """Las tesis de `recepcion_mx` que entran, por prioridad y no por el orden
    del archivo (revisión crítica del pilar, 25-sep-2026: con la pregunta de
    David el recorte por orden dejaba sólo registros de 2011-2013 y fuera las
    P./J. 20/2014, 21/2014, 2/2022 y 2/2025).

    Primero lo FIJO: el estado actual y, con la pregunta amplia, el núcleo
    mexicano. Después lo que piden la sonda y los temas, POR TURNOS (una de
    cada tema por vuelta, cada tema en su orden de prioridad). Al final, lo que
    más puntúa: relevancia a la pregunta (idf) + vigencia + fuerza. Y las
    PAREJAS: si entra una abandonada entra su sustituta y al revés, por encima
    del tope (MAX_PAREJAS), para que el modelo nunca vea una sin la otra.

    Devuelve {registros, recortar, parejas, abandonos}: `recortar` dice qué
    puede quitar traer_linea si el bloque pasa del presupuesto (lo que entró
    por puntuación, las parejas que no son de un abandono y lo que la sonda y
    los temas trajeron de la segunda vuelta en adelante; nunca lo fijo ni la
    sustituta de una abandonada que no se puede quitar); `parejas` dice quién
    trajo a cada pareja y `abandonos`, abandonada → sustituta de las que
    entraron juntas."""
    f = figuras().get(fid) or {}
    crono = _crono_por_id(fid)
    rec = {str(r["registro"]): r for r in f.get("recepcion_mx") or [] if r.get("en_acervo")}
    temas_mx = f.get("temas_mx") or {}
    vacio: Dict[str, Any] = dict(registros=[], parejas={}, abandonos={},
                                 recortar=dict(puntos=[], tardios=[], parejas=[]))
    # Primero el estado actual: con «¿cuál es la postura ACTUAL…?» la P./J.
    # 20/2014, la 21/2014, la 2/2022, la 64/2014 y la 2/2025 no pueden quedar
    # detrás de las dieciséis de un tema.
    actual = "actual" in reales and (mexico or tema_primero or solo_mx)
    sonda: List[str] = []
    for i in ids:
        if i in rec:
            sonda.append(i)
        elif i in crono and str(crono[i].get("registro") or "") in rec:
            sonda.append(str(crono[i]["registro"]))
    sonda = list(dict.fromkeys(sonda))
    listas_tema = [[str(x) for x in (temas_mx.get(tid) or {}).get("registros") or [] if str(x) in rec]
                   for tid in temas]
    # El núcleo mexicano y la recepción entera sólo cuando se pregunta por la
    # LÍNEA (no por un tema) con México en la pregunta o en modo solo_mx;
    # «¿dónde nace…?» o «¿quiénes están obligados hoy?» traen unas pocas
    # tesis de la recepción, no la P./J. 2/2022 (medido: sin este corte la
    # pregunta interamericana pasaba de 8 mil tokens).
    linea = bool(reales) and not tema_primero
    amplia = linea and (mexico or solo_mx)
    nucleo = [str(x) for x in f.get("nucleo_mx") or [] if str(x) in rec] if amplia else []
    # Lo del estado actual no compite: entra entero (con «¿cuál es la postura
    # actual de la SCJN sobre el control difuso?» la P./J. 21/2014 y la 2/2025
    # quedaban detrás de las tesis que dicen «difuso» en el rubro). Y el NÚCLEO
    # TAMPOCO (revisión adversarial, 26-sep-2026): con el hilo real de la
    # pregunta de David sobre la línea de la SCJN, 3×idf de raíces genéricas
    # («criter», «jurisp», «mexico») subía la P. LXVI/2011 superada a 28 puntos
    # y dejaba la P./J. 20/2014 en el lugar 10 y la P. LXVII/2011 fuera; el +3
    # del núcleo no alcanzaba. La docstring prometía justo lo contrario.
    fijos = [str(x) for x in (f.get("fase_actual") or {}).get("registros") or [] if str(x) in rec] if actual else []
    fijos = list(dict.fromkeys(fijos + nucleo))
    cands = set(fijos) | set(sonda) | {x for lista in listas_tema for x in lista}
    if linea:
        # La recepción de la SCJN; los Plenos Regionales y los colegiados
        # entran sólo si un tema, el estado actual o la sonda los piden.
        cands |= {k for k, r in rec.items() if r.get("instancia") in ("Pleno", "Primera Sala", "Segunda Sala")}
    if not cands:
        return vacio
    tope = MAX_TESIS if (temas or ids) else MAX_TESIS_LINEA if amplia else 4
    pedidos = set(sonda) | {x for lista in listas_tema for x in lista}

    def prio(reg: str) -> float:
        r = rec[reg]
        s = 3.0 * _relevancia(fid, q, _texto_tesis(r, crono.get(r.get("crono_id") or "")))
        s += _PESO_VIGENCIA.get(r.get("vigencia") or "vigente", 0.0) + _PESO_FUERZA_TESIS.get(r.get("fuerza"), 0.0)
        if reg in nucleo:
            s += 3.0
        fecha = str(r.get("fecha_publicacion") or "")
        if "actual" in reales and fecha >= "2014":
            s += 1.0
        if "origen" in reales and fecha and fecha <= "2011-12-31":
            # El origen en México es la recepción del Varios 912/2010 (la P.
            # LXVII/2011 es la tesis madre), no la jurisprudencia posterior.
            s += 2.5 if r.get("relacion") == "recibe" else 1.0
        # Plenos Regionales y colegiados: sólo si un tema los pide o la
        # pregunta los toca de verdad (en la línea general, la IX.P. J/2 P
        # subía por decir «línea jurisprudencial» en su rubro).
        if r.get("instancia") in ("Plenos Regionales", "Tribunales Colegiados de Circuito") and reg not in pedidos:
            s -= 4.0
        return s

    # En empate, primero el núcleo mexicano en su orden (la P. LXVII/2011, tesis
    # madre del Varios 912/2010, antes que sus hermanas del mismo mes): con las
    # fechas completas de la P. LXV/2011 y la P. LXXI/2011 (26-sep-2026) el
    # empate por registro la dejaba fuera de «¿dónde nace…?».
    orden_nucleo = {str(x): i for i, x in enumerate(f.get("nucleo_mx") or [])}

    def por_prio(xs: Iterable[str]) -> List[str]:
        return sorted(xs, key=lambda reg: (-prio(reg), orden_nucleo.get(reg, len(orden_nucleo)), reg))

    # Lo que piden la sonda y los temas, por turnos (la sonda, en su orden de
    # score; cada tema, por prioridad; lo fijo ya está y no gasta turno).
    turnos = _por_turnos([[x for x in sonda if x not in fijos]]
                         + [por_prio(x for x in lista if x not in fijos) for lista in listas_tema])
    forz = [x for x, _, _ in turnos]
    generales = por_prio(cands - set(fijos) - set(forz))
    libres = max(0, tope - len(fijos))
    entran = (forz + generales)[:libres]
    elegidos = fijos + entran
    parejas: List[Tuple[int, str]] = []
    de: Dict[str, Dict[str, Any]] = {}
    for reg in elegidos:
        r = rec[reg]
        otros = ([str(r["reemplazo"])] if r.get("reemplazo") else []) + [str(x) for x in r.get("sustituye") or []]
        for o in otros:
            if o in rec and o not in elegidos:
                abandono = "abandonada" in (r.get("vigencia"), rec[o].get("vigencia"))
                # Un abandono va SIEMPRE en pareja (el error de David); una
                # superación de contenido, sólo si la pidió un tema o la sonda.
                if abandono or temas or ids:
                    if o not in de:
                        parejas.append((0 if abandono else 1, o))
                        de[o] = dict(de=[], abandono=abandono)
                    de[o]["de"].append(reg)
                    de[o]["abandono"] = de[o]["abandono"] or abandono
    parejas.sort()
    entran_parejas = [o for _, o in parejas[:MAX_PAREJAS]]
    de = {o: de[o] for o in entran_parejas}
    # Qué se quita primero si el bloque no cabe: lo que entró por puntuación
    # (lo último, primero), las parejas que no son de un abandono y, al final,
    # lo que la sonda y los temas trajeron de la segunda vuelta en adelante
    # (la última vuelta primero: el primer tema que nombra la pregunta es el
    # último en perder).
    vuelta = {x: (v, k) for x, v, k in turnos}
    por_puntos = [x for x in entran if x not in vuelta]
    tardios = [(x,) + vuelta[x] for x in entran if x in vuelta and vuelta[x][0] >= 1]
    recortar_parejas = [o for o in entran_parejas if not de[o]["abandono"]][::-1]
    # LA PAREJA DE UN ABANDONO, VENGA DE DONDE VENGA (revisión del 26-sep-2026,
    # hallazgo 14.3). `parejas` sólo sabe de las que trajo el mecanismo de
    # arriba; cuando la abandonada y su sustituta llegan las dos por la lista
    # de un tema, nadie las ataba. Medido con Qdrant real: en «¿pueden los
    # tribunales colegiados ejercer control difuso sobre la prisión preventiva
    # oficiosa y el arraigo…?» la P. X/2015 (2009817) es la primera vuelta del
    # tema —protegida— y la P./J. 2/2022 (2024159) la tercera; con un
    # presupuesto de 3,500 se iba la sustituta y quedaba la abandonada sola, el
    # error de David. Si la abandonada no se puede recortar, su sustituta
    # tampoco; si las dos se pueden, `abandonos` hace que la sustituta se lleve
    # a la abandonada (ver _al_presupuesto).
    todos_reg = elegidos + entran_parejas
    abandonos = {x: str(rec[x]["reemplazo"]) for x in todos_reg
                 if rec[x].get("vigencia") == "abandonada" and str(rec[x].get("reemplazo") or "") in todos_reg}
    recortables_reg = set(por_puntos) | {z[0] for z in tardios} | set(recortar_parejas)
    # La abandonada que entró como pareja se va con quien la trajo (ver
    # _al_presupuesto): cuenta como recortable si todos los que la trajeron lo
    # son, y entonces no blinda a su sustituta.
    recortables_reg |= {o for o in entran_parejas if de[o]["abandono"]
                        and all(b in recortables_reg for b in de[o]["de"])}
    blindadas = {abandonos[x] for x in abandonos if x not in recortables_reg}
    return dict(registros=todos_reg, parejas=de, abandonos=abandonos,
                recortar=dict(puntos=[x for x in por_puntos[::-1] if x not in blindadas],
                              tardios=[z for z in tardios[::-1] if z[0] not in blindadas],
                              parejas=[x for x in recortar_parejas if x not in blindadas]))


def _elegir_cronologia(fid: str, q: frozenset, reales: Sequence[str], temas: Sequence[str], mexico: bool,
                       tema_primero: bool, ids: Sequence[str], solo_mx: bool) -> Dict[str, Any]:
    """Las entradas de la cronología que van en <cronologia>: lo que NO es de la
    Corte IDH (eso va en <hitos>) ni una tesis del acervo (va en <tesis>), o sea
    resoluciones y acuerdos de la SCJN, votos, proyectos, reformas y leyes,
    informes de la CIDH y de la ONU, doctrina y tesis que no están en el acervo.
    Primero lo que piden la sonda, los temas y el estado actual, POR TURNOS
    (revisión adversarial, 26-sep-2026: concatenados, el primer tema del
    archivo se quedaba el cupo); después, si la pregunta es de la línea, lo que
    más puntúa, con topes por clase.

    Desde el 26-sep-2026 (hallazgo 14.2) lo del estado actual va fijo y
    primero, fuera de los turnos, y los temas y la sonda se reparten sólo lo
    que les queda.

    Devuelve {cronologia, recortar}: si el bloque no cabe se quita primero lo
    que entró por puntuación y después lo que la sonda y los temas trajeron de
    la segunda vuelta en adelante; lo del estado actual, nunca."""
    f = figuras().get(fid) or {}
    crono = cronologia(fid)
    rec = {str(r["registro"]) for r in f.get("recepcion_mx") or [] if r.get("en_acervo")}

    def candidata(e: Dict[str, Any]) -> bool:
        if e.get("ref_hito") or e.get("tipo") in _TIPOS_CORTE_IDH:
            return False
        if e.get("registro") and str(e["registro"]) in rec:
            return False
        return not solo_mx or e.get("tipo") in _TIPOS_SOLO_MX

    cands = {e["id"]: e for e in crono if candidata(e)}
    # LO DEL ESTADO ACTUAL VA FIJO Y NO GASTA TURNO DE NADIE (revisión del
    # 26-sep-2026, hallazgo 14.2). Iba como una lista más en _por_turnos y el
    # tope de MAX_CRONO + 2 contaba todo junto: en «Traza la línea…» y en
    # «¿Qué posturas serias hay para inaplicar restricciones constitucionales
    # como la prisión preventiva oficiosa?», los dos primeros ids del tema de
    # prisión preventiva (la reforma del 31-dic-2024 y el AG 2/2024) ya los
    # ponía el estado actual y aun así le cobraban sus dos turnos: el tema se
    # quedaba sin ninguna entrada propia (fuera el art. 166 LA, la CC 3/2026 y
    # el proyecto de Pardo) y el voto de Cossío en la CT 293/2011 ni se
    # consideraba. Como en _elegir_registros, lo fijo va primero y las listas
    # de los temas y la sonda compiten sólo con lo que les queda. Sin tope de
    # cantidad: las listas de los temas son curadas (seis entradas como mucho)
    # y lo de la segunda vuelta en adelante lo quita el presupuesto, que ahora
    # se aplica (ver _al_presupuesto).
    estado: List[str] = []
    if "actual" in reales and (mexico or tema_primero or solo_mx):
        estado = [i for i in (f.get("fase_actual") or {}).get("cronologia") or [] if i in cands]
    fuentes: List[List[str]] = [[i for i in ids if i in cands and i not in estado]]
    for tid in temas:
        fuentes.append([i for i in ((f.get("temas_mx") or {}).get(tid) or {}).get("cronologia") or []
                        if i in cands and i not in estado])
    turnos = _por_turnos(fuentes)
    elegidos = estado + [i for i, _, _ in turnos]
    # Lo del estado actual no se recorta (como sus hitos y sus tesis); de la
    # sonda y los temas, sólo lo de la segunda vuelta en adelante.
    tardios = [(i, v, k) for i, v, k in turnos if v >= 1]
    por_puntos: List[str] = []
    # Por puntuación sólo cuando se pregunta por la LÍNEA con México (o en modo
    # solo_mx): un tema trae lo suyo y nada más (medido el 25-sep-2026: con
    # México en la pregunta de los Colegiados entraban la «supremacía
    # convencional» de Ferrer y un informe de la CIDH que no venían al caso).
    if reales and not tema_primero and (mexico or solo_mx) and len(elegidos) < MAX_CRONO:
        tramos_q = set()
        for x in list(reales) + (["mexico"] if mexico else []):
            tramos_q |= set(_TRAMOS_FASE.get(x, ()))
        tramos_q |= {_TRAMO_TEMA[t] for t in temas if t in _TRAMO_TEMA}

        def rel(e: Dict[str, Any]) -> float:
            return 3.0 * _relevancia(fid, q, _texto_entrada(e))

        def prio(e: Dict[str, Any]) -> float:
            s = rel(e) + (2.0 if e.get("tramo") in tramos_q else 0.0)
            s += _PESO_FUERZA.get(e.get("fuerza"), 0.0) + 0.5 * _PESO_VIGENCIA.get(e.get("vigencia"), 0.0)
            if "actual" in reales and str(e.get("fecha_resolucion") or e.get("fecha_publicacion") or "") >= "2022":
                s += 1.0
            return s

        clases: Counter = Counter()
        for e in sorted((c for i, c in cands.items() if i not in elegidos), key=lambda e: (-prio(e), e.get("orden") or 0)):
            if len(elegidos) >= MAX_CRONO or prio(e) < 4.0:
                break
            # Fuera de los tramos de la pregunta, sólo lo que la toca de verdad.
            if e.get("tramo") not in tramos_q and rel(e) < 4.5:
                continue
            clase = "recomendacion" if e.get("tipo") in ("cidh", "onu") else e.get("tipo")
            if clase in _LIMITE_CLASE and clases[clase] >= _LIMITE_CLASE[clase]:
                continue
            clases[clase] += 1
            elegidos.append(e["id"])
            por_puntos.append(e["id"])
    return dict(cronologia=sorted(elegidos, key=lambda i: cands[i].get("orden") or 0),
                recortar=dict(puntos=por_puntos[::-1], tardios=tardios[::-1]))


def seleccionar(fid: str, fases: Sequence[str], pregunta: Optional[str] = None, alcance: Optional[str] = "completa",
                ids: Sequence[str] = ()) -> Dict[str, Any]:
    """Qué se trae para (figura, fases): hitos de la Corte IDH, registros de la
    SCJN, entradas de la cronología, artículos constitucionales y
    supervisiones. Sin red; se puede probar solo.

    Con presupuesto (medido con cl100k el 25-sep-2026: ~230 tokens por hito):
    el núcleo del origen cuando se pide el origen, la evolución, el concepto
    o un eje (no con «¿cuál es el estado actual…?» a secas); Radilla cuando
    además se pregunta por México; los ejes pedidos, enteros; un cupo por fase
    —«actual» toma lo más reciente, las demás lo que más aporta según la
    arista— con al menos MIN_POR_FASE garantizados a cada fase que el núcleo o
    el estado actual no cubren, hasta MAX_HITOS (MAX_HITOS_MEXICO si entra
    México, para dejar sitio a las tesis y la cronología). Después, orden
    cronológico.

    `pregunta` ordena por relevancia las tesis y la cronología; `alcance`
    «solo_mx» deja fuera todo lo interamericano (ver alcance_por_fuentes);
    `ids` son hitos, registros o entradas que la sonda semántica dio por
    relevantes y entran por delante.

    `recortables` es el orden en que traer_linea quita piezas si el bloque
    pasa de presupuesto(sel): [(«cronologia»|«registro»|«hito»|
    «supervision»|«tensiones», id)], lo más prescindible primero; `abandonos`
    (abandonada → sustituta) ata a las dos en el recorte."""
    f = figuras().get(fid) or {}
    por_llave = _hitos_por_llave(fid)
    temas = [x[5:] for x in fases if x.startswith("tema:")]
    ejes = [x[4:] for x in fases if x.startswith("eje:")]
    mexico = "mexico" in fases
    tema_primero = "tema_primero" in fases
    reales = [x for x in fases if x in _FASES_HITO]
    solo_mx = alcance == "solo_mx"
    ids = [str(i) for i in ids or ()]
    q = _raices(pregunta or "")

    elegidos: Dict[str, int] = {}          # llave → prioridad (menor = antes)
    protegidos: set = set()                 # lo que el presupuesto nunca quita
    de_tema: List[List[str]] = []           # los hitos que trae cada tema
    por_fase: Dict[str, List[str]] = {}     # lo que el cupo tomó para cada fase
    constitucion: List[str] = []
    supervisiones: List[Dict[str, Any]] = []
    advertencias: List[str] = []
    todos = [h for h in f.get("hitos") or [] if not h.get("solo_por_tema")]
    actual_mx = "actual" in reales and (mexico or tema_primero)

    def fase_de(h):
        return (h.get("fase") or "").lower()

    if reales and not solo_mx:
        # El núcleo del origen, cuando se pide la línea o uno de sus ejes; no
        # con «¿cuál es el estado actual…?» a secas (eran ~1,200 tokens de
        # origen en una pregunta que no lo pide; 25-sep-2026).
        if not tema_primero and (ejes or set(reales) & {"origen", "evolucion", "concepto"}):
            for ll in NUCLEO:
                if ll in por_llave:
                    elegidos[ll] = -1
                    protegidos.add(ll)
        # Y con México, Radilla ¶339-341: cómo llegó el control a México
        # (revisión adversarial, 26-sep-2026; ver RADILLA).
        if mexico and not tema_primero and set(reales) & {"origen", "evolucion", "concepto"}:
            for ll in RADILLA:
                if ll in por_llave:
                    elegidos[ll] = min(elegidos.get(ll, 0), 0)
                    protegidos.add(ll)
        for eje in ejes:
            rx_eje = _RX_EJES.get(eje)
            for h in todos:
                # Del eje, todos; de otra fase, sólo lo que AMPLÍA el eje según
                # su aporte: Santo Domingo ¶142 está en «complementariedad»
                # pero es el de «todas las autoridades y órganos» (oro 5).
                if fase_de(h) == eje or (h.get("arista") in ("amplía", "origina") and rx_eje is not None
                                         and rx_eje.search(_plegar(h.get("aporte") or ""))):
                    elegidos.setdefault(h["llave"], 0)
                    protegidos.add(h["llave"])
        tope = MAX_HITOS_MEXICO if mexico else MAX_HITOS
        libres = max(0, tope - len(elegidos))
        cupo = max(MIN_POR_FASE, libres // max(1, len(reales)))
        fin_origen = max((por_llave[ll].get("fecha") or "" for ll in NUCLEO if ll in elegidos), default="")
        # Con «tema_primero» el tema manda: no se llena la fase con lo más
        # reciente de la Corte IDH en otros países (la postura ACTUAL de la
        # SCJN sobre el control difuso no pide a Huilcamán ni la OC-32).
        for x in ([] if tema_primero else reales):
            quiere = set(_FASES_HITO[x])
            if not mexico:
                quiere -= {"contenido (méxico)"}
            cands = [h for h in todos if fase_de(h) in quiere and h["llave"] not in elegidos
                     # Las reiteraciones pesan poco… salvo en el estado actual,
                     # donde que la Corte siga diciéndolo en 2024-2025 ES la noticia.
                     and (h.get("arista") != "reitera" or x == "actual")]
            if x == "evolucion" and fin_origen:
                # Con el núcleo dentro, la evolución es lo de DESPUÉS de
                # Cesados (26-sep-2026): el voto de García Ramírez en Cesados
                # es del mismo día y ocupaba uno de los dos lugares.
                cands = [h for h in cands if (h.get("fecha") or "") > fin_origen]
            if x == "actual":
                cands.sort(key=lambda h: h.get("fecha") or "", reverse=True)
                if not mexico:     # sin México en la pregunta, primero lo interamericano
                    cands.sort(key=lambda h: "(méxico)" in fase_de(h))
            else:
                cands.sort(key=lambda h: (_PESO_ARISTA.get(h.get("arista"), 5), h.get("fecha") or ""))
            por_fase[x] = [h["llave"] for h in cands[:cupo]]
            for h in cands[:cupo]:
                elegidos.setdefault(h["llave"], 1)
        if mexico:
            constitucion += list(_CONSTITUCION_MEXICO)

    if not solo_mx:
        # El estado a sep-2026 con México: Tzompaxtle ¶118 y García Rodríguez
        # ¶176, 301 y 303 entran siempre (la crítica del pilar los encontró
        # fuera: la fase «actual» sólo miraba lo más reciente).
        if actual_mx:
            for ll in (f.get("fase_actual") or {}).get("hitos") or []:
                if ll in por_llave:
                    elegidos[ll] = min(elegidos.get(ll, 0), 0)
                    protegidos.add(ll)
        for tid in temas:
            t = (f.get("temas_mx") or {}).get(tid) or {}
            de_tema.append([ll for ll in t.get("hitos") or [] if ll in por_llave])
            for ll in t.get("hitos") or []:
                if ll in por_llave:
                    elegidos[ll] = min(elegidos.get(ll, 0), 0)
            for x in t.get("extra") or []:
                elegidos.setdefault(x["llave"], 1)
            constitucion = list(t.get("constitucion") or []) + constitucion
            for s in t.get("supervision") or []:
                if s.get("doc_id") not in {x.get("doc_id") for x in supervisiones}:
                    supervisiones.append(s)
        if actual_mx:
            # Y las supervisiones del 26-nov-2024, que dejaron abierto el punto.
            for tid in ("prision_preventiva_oficiosa", "arraigo"):
                for s in ((f.get("temas_mx") or {}).get(tid) or {}).get("supervision") or []:
                    if s.get("doc_id") not in {x.get("doc_id") for x in supervisiones}:
                        supervisiones.append(s)
        for x in ids:
            if x in por_llave:
                elegidos[x] = min(elegidos.get(x, 0), 0)
                protegidos.add(x)
        # Los hitos esenciales de cada tema pedido no se recortan (hallazgo
        # 14.1, 26-sep-2026; ver _esenciales_tema). Se miden contra lo fijo
        # ya elegido, para no proteger un resolutivo que repite la orden de un
        # párrafo de su misma sentencia.
        fijos_ya = [ll for ll in elegidos if elegidos[ll] <= 0]
        for lista in de_tema:
            protegidos |= set(_esenciales_tema(fid, lista, fijos_ya))
    for tid in temas:
        t = (f.get("temas_mx") or {}).get(tid) or {}
        if t.get("advertencia"):
            advertencias.append(t["advertencia"])

    # El tope recorta lo de cupo, nunca el núcleo, Radilla, los ejes, los
    # temas ni el estado actual. A cada fase pedida que no cubren ya el núcleo
    # (el origen) o los hitos fijos del estado actual con México le garantiza
    # MIN_POR_FASE de su cupo, también con México («¿quiénes están obligados
    # HOY?» llena el eje de sujetos y aun así trae lo más reciente; la línea
    # con México ya no salta de 2006 a 2022; revisión adversarial, 26-sep-2026).
    tope = MAX_HITOS_MEXICO if mexico else MAX_HITOS
    fijos = [ll for ll in elegidos if elegidos[ll] <= 0]
    cubiertas = ({"origen"} if any(ll in fijos for ll in NUCLEO) else set()) | ({"actual"} if actual_mx else set())
    garantia: List[str] = []
    for x in ([] if tema_primero else reales):
        if x not in cubiertas:
            garantia += [ll for ll in por_fase.get(x, []) if ll not in fijos and ll not in garantia][:MIN_POR_FASE]
    protegidos |= set(garantia)
    resto = sorted((ll for ll in elegidos if elegidos[ll] > 0 and ll not in garantia),
                   key=lambda ll: (elegidos[ll], por_llave[ll].get("fecha") or "", ll))
    extra = resto[:max(tope - len(fijos) - len(garantia), 0)]
    orden = fijos + garantia + extra
    hitos = sorted((por_llave[ll] for ll in orden), key=lambda h: (h.get("fecha") or "", _orden_llave(h["llave"])))
    registros = _elegir_registros(fid, q, reales, temas, mexico, tema_primero, ids, solo_mx)
    crono = _elegir_cronologia(fid, q, reales, temas, mexico, tema_primero, ids, solo_mx)
    # Qué se quita primero si el bloque pasa del presupuesto: lo que entró por
    # puntuación (cronología, tesis, hitos de cupo), las parejas que no son de
    # un abandono, el extracto de la supervisión, lo que los temas trajeron de
    # la segunda vuelta en adelante, los hitos de un tema que no son
    # esenciales y, al final, las tensiones. Nunca lo protegido: el núcleo,
    # Radilla, el estado actual, lo garantizado a cada fase y los hitos
    # ESENCIALES de cada tema (_esenciales_tema).
    #
    # EL ORDEN CAMBIÓ el 26-sep-2026 (reverificación, hallazgo 14.1). Antes
    # las tensiones y TODOS los hitos de un tema se iban antes que cualquier
    # tesis o entrada de la segunda vuelta. Medido con Qdrant real y el tope de
    # un tema (6,500): «prisión preventiva oficiosa y restricciones
    # constitucionales expresas en el parámetro de regularidad» pasaba de 9
    # hitos de la Corte IDH a 0, y «¿pueden los tribunales colegiados ejercer
    # control difuso sobre la prisión preventiva oficiosa y el arraigo…?», a 0,
    # mientras seguían dentro la 2a. CXXVIII/2015 (2010428) y la 2a./J.
    # 119/2014 (2007932), de segunda vuelta. Sin García Rodríguez ¶301/¶303 ni los
    # resolutivos que ORDENAN adecuar, el lado interamericano de la respuesta
    # se contaba de memoria. Y las tensiones de los temas pedidos son justo
    # «las posturas serias» que pregunta David: se quitan al último.
    hitos_tema = [ll for ll, _, _ in _por_turnos(de_tema) if ll in orden and ll not in protegidos]
    con_extracto = bool(set(temas) & {"prision_preventiva_oficiosa", "arraigo"}) and bool(supervisiones)
    recortables = ([("cronologia", i) for i in crono["recortar"]["puntos"]]
                   + [("registro", r) for r in registros["recortar"]["puntos"]]
                   + [("hito", ll) for ll in extra[::-1] if ll not in protegidos]
                   + [("registro", r) for r in registros["recortar"]["parejas"]]
                   + ([("supervision", "extracto")] if con_extracto else [])
                   # Lo tardío de los temas, la última vuelta primero y, en la
                   # misma vuelta, el último tema nombrado primero.
                   + [(t, x) for t, x, _, _ in sorted(
                       [("cronologia",) + tuple(z) for z in crono["recortar"]["tardios"]]
                       + [("registro",) + tuple(z) for z in registros["recortar"]["tardios"]],
                       key=lambda z: (-z[2], -z[3], z[0] == "registro"))]
                   + [("hito", ll) for ll in hitos_tema[::-1]]
                   + [("tensiones", 1), ("tensiones", 0)])
    return dict(hitos=hitos, registros=list(dict.fromkeys(registros["registros"])), cronologia=crono["cronologia"],
                constitucion=[] if solo_mx else list(dict.fromkeys(constitucion)), supervisiones=supervisiones,
                advertencias=advertencias, temas=temas, fases=reales, ejes=ejes, mexico=mexico,
                alcance=alcance or "completa", tema_primero=tema_primero,
                actual_mx=actual_mx or (solo_mx and "actual" in reales), extracto_supervision=con_extracto,
                parejas=registros["parejas"], abandonos=registros["abandonos"], recortables=recortables)


def _orden_llave(ll: str) -> Tuple[int, int]:
    """Dentro del mismo día: la sentencia antes que sus votos, y por párrafo."""
    seg = ll.split("|")[1] if ll.count("|") >= 2 else ""
    num = ll.rsplit("|", 1)[-1]
    return (0 if seg in ("s", "c") else 1 if seg.startswith("r") else 2,
            int(num) if num.isdigit() else 0)


# ═══════════════════════════════════════════════════════════════ el contrato

def _nombre_autor(slug: Optional[str]) -> Optional[str]:
    """«garcia-ramirez» → «García Ramírez»; los votos conjuntos van con «+»."""
    if not slug:
        return None
    try:
        import coidh_catalogo
        autores = coidh_catalogo.catalogo()["data"]["autores"]
    except Exception:
        autores = {}
    nombres = []
    for s in str(slug).split("+"):
        s = s.split("@")[0]
        nombres.append((autores.get(s) or {}).get("nombre") or s.replace("-", " ").title())
    return " y ".join(nombres)


def _serie_rotulo(serie: Optional[str], num: Optional[int]) -> Optional[str]:
    if serie in ("C", "A") and num:
        return f"Serie {serie} No. {num}"
    return None


def _origen(doc_id: str, caso: Optional[str], fecha: Optional[str]) -> str:
    """Cómo se rotula la fuente en la app («Disposición legal» si faltara)."""
    caso = caso or doc_id
    if doc_id.startswith("A-"):
        return f"Corte IDH. Opinión Consultiva {caso.split(' · ')[0]}"
    if doc_id.startswith("SS-"):
        base = caso.replace(" (supervisión de cumplimiento)", "")
        return f"Corte IDH. Caso {base}. Supervisión de cumplimiento ({fecha_larga(fecha)})"
    return f"Corte IDH. Caso {caso}"


def contrato(pid: Any, pl: Dict[str, Any], score: float = 1.0, rol: str = "pedido") -> Dict[str, Any]:
    """Un punto de la colección `coidh` → los campos del contrato del frontend
    (y del resto de SearchResult). SIN «#page» en ninguna URL: la página viaja
    aparte (con #page pegado, la app Android dejaba de reconocer el PDF, plan
    §3.8). `pdf_url` es la copia de legal-docs si la hay (`pdf_de`, 25-sep-2026)
    y `url_oficial`, la de corteidh.or.cr. `parrafo` va como texto;
    `voto_autor`, con nombre."""
    url = (pl.get("url_oficial") or pl.get("pdf_url") or "").split("#")[0] or None
    pdf_url, sha1 = pdf_de(url, pl.get("pdf_sha1"))
    doc_id = str(pl.get("doc_id") or "")
    parrafo = pl.get("parrafo")
    return dict(
        id=str(pid), score=score,
        texto=pl.get("texto") or pl.get("texto_raw") or "",
        ref=pl.get("ref"),
        origen=_origen(doc_id, pl.get("caso"), pl.get("fecha")),
        jurisdiccion="Corte Interamericana de Derechos Humanos",
        entidad=None, silo=SILO, pdf_url=pdf_url,
        tipo=pl.get("tipo"), url_oficial=url,
        pagina=int(pl["pagina"]) if isinstance(pl.get("pagina"), (int, float)) else None,
        parrafo=str(parrafo) if parrafo is not None else None,
        seg=pl.get("seg"), voto_autor=_nombre_autor(pl.get("voto_autor")),
        caso=pl.get("caso"), serie=_serie_rotulo(pl.get("serie"), pl.get("serie_num")),
        fecha=pl.get("fecha"), ancla=pl.get("ancla"), cita_canonica=pl.get("cita_canonica"),
        llave=pl.get("llave"), rol_coidh=rol, pdf_sha1=sha1,
        # internos (no viajan al frontend)
        _doc_id=doc_id, _orden=pl.get("orden"), _sub=pl.get("sub") or 0,
    )


def _tipo_de_doc(doc_id: str, seg: str) -> str:
    if seg == "voto":
        return "voto_coidh"
    if seg == "resolutivos":
        return "resolutivo_coidh"
    if doc_id.startswith("A-"):
        return "oc_coidh"
    if doc_id.startswith("SS-"):
        return "supervision_coidh"
    return "sentencia_coidh"


def ficha_catalogo(doc_id: str, cita_canonica: Optional[str] = None, parrafo: Optional[Any] = None,
                   seg: str = "sentencia", voto_autor: Optional[str] = None, llave: Optional[str] = None,
                   nota: Optional[str] = None, rol: str = "ficha") -> Optional[Dict[str, Any]]:
    """La resolución del catálogo SIN su texto: cita oficial y URL oficial.

    Es el respaldo del plan §3.6 («el visor abre la sentencia oficial para las
    597, no sólo para las del piloto»). No lleva ni una palabra de la
    sentencia: lo que no está ingerido no se parafrasea."""
    try:
        import coidh_catalogo
        d = coidh_catalogo.documento(doc_id)
    except Exception:
        d = None
    if not d:
        return None
    caso = (f"{d['oc']} · {d['nombre']}" if d.get("clase") == "OC" and d.get("oc")
            else f"{d['nombre']} Vs. {d['estado']}" if d.get("estado") else d.get("nombre"))
    serie = _serie_rotulo(d.get("serie"), d.get("serie_num"))
    cita = cita_canonica or d.get("cita")
    cab = " | ".join(x for x in ("Corte IDH", caso, d.get("fecha"), serie) if x)
    texto = (f"[{cab}]\nFicha del catálogo oficial de la Corte IDH: {cita}\n"
             "El texto de esta resolución no está ingerido en Iurexia; se conocen su cita y su "
             "enlace oficial, no su contenido." + (f"\n{nota}" if nota else ""))
    # La copia de legal-docs, si la hay (25-sep-2026): la mayoría de las fichas
    # son de resoluciones sin copia y siguen abriendo la URL oficial.
    oficial = (d.get("url_oficial") or "").split("#")[0] or None
    pdf_url, sha1 = pdf_de(oficial)
    return dict(
        id=ficha_id(doc_id), score=1.0, texto=texto,
        ref=("Ficha del catálogo" if parrafo is None else f"Párr. {parrafo} (no ingerido)"),
        origen=_origen(doc_id, caso, d.get("fecha")),
        jurisdiccion="Corte Interamericana de Derechos Humanos", entidad=None, silo=SILO,
        pdf_url=pdf_url, tipo=_tipo_de_doc(doc_id, seg), url_oficial=oficial,
        pagina=None, parrafo=str(parrafo) if parrafo is not None else None, seg=seg,
        voto_autor=voto_autor, caso=caso, serie=serie, fecha=d.get("fecha"), ancla=None,
        cita_canonica=cita, llave=llave or doc_id, rol_coidh=rol, pdf_sha1=sha1,
        _doc_id=doc_id, _orden=None, _sub=0,
    )


def ficha_hito(h: Dict[str, Any], rol: str = "hito") -> Dict[str, Any]:
    """Un hito de la ficha curada que no está en la colección (fuera del piloto
    o sin cotejar). Su extracto SÍ es literal —verificado en su página del PDF
    oficial en F0—, así que va en el texto y se puede citar entre comillas."""
    ll = h["llave"]
    doc_id = h.get("doc_id") or ll.split("|")[0]
    seg = {"considerando": "considerandos"}.get(h.get("seg"), h.get("seg"))
    if not seg:
        seg = _SEG_LLAVE.get(ll.split("|")[1][:1], "voto") if ll.count("|") >= 2 else "sentencia"
    # Los `extra` de temas_mx (C-220 ¶233, C-567 ¶267) traen sólo llave,
    # página, caso, fecha y extracto: sin esto su ficha salía con parrafo,
    # serie y cita en null y el <cita> del bloque vacío (revisión del
    # 25-sep-2026). Se completan de la llave y del catálogo, sin inventar.
    h = dict(h)
    num_ll = ll.split("|")[2] if ll.count("|") >= 2 else ""
    if h.get("parrafo") is None and num_ll.isdigit():
        h["parrafo"] = int(num_ll)
    m_serie = re.fullmatch(r"([AC])-(\d+)", doc_id)
    if not h.get("serie") and m_serie:
        h["serie"], h["serie_num"] = m_serie.group(1), int(m_serie.group(2))
    if not h.get("cita_canonica"):
        try:
            import coidh_catalogo
            d_cat = coidh_catalogo.documento(doc_id)
            if d_cat and d_cat.get("cita"):
                tramo = ll.split("|")[1] if ll.count("|") >= 2 else ""
                slug = tramo[2:] if tramo.startswith("v:") else None
                h["cita_canonica"] = coidh_catalogo._cita_canonica(
                    d_cat, seg, [h["parrafo"]] if isinstance(h.get("parrafo"), int) else [], slug)
        except Exception:
            pass
    serie = _serie_rotulo(h.get("serie"), h.get("serie_num"))
    url = (h.get("url_oficial") or "").split("#")[0] or None
    # Con copia en legal-docs (25-sep-2026), el visor abre la copia en la
    # página del hito; la cita sigue apuntando a corteidh.or.cr.
    pdf_url, sha1 = pdf_de(url, h.get("pdf_sha1"))
    ubic = h.get("ubicacion") or (f"párr. {h['parrafo']}" if h.get("parrafo") is not None else "")
    cab = " | ".join(x for x in ("Corte IDH", h.get("caso"), h.get("fecha"), serie, ubic) if x)
    texto = (f"[{cab}]\nHito de la línea curada; el texto completo de esta resolución no está "
             "ingerido en Iurexia."
             + (f"\nExtracto verificado en la pág. {h['pagina']} del PDF oficial: «{h['extracto']}»"
                if h.get("extracto") and h.get("verificado") else ""))
    return dict(
        id=ficha_id(ll), score=1.0, texto=texto,
        ref=(h.get("ubicacion") or (f"Párr. {h['parrafo']}" if h.get("parrafo") is not None else "Hito")),
        origen=_origen(doc_id, h.get("caso"), h.get("fecha")),
        jurisdiccion="Corte Interamericana de Derechos Humanos", entidad=None, silo=SILO,
        pdf_url=pdf_url, tipo=_tipo_de_doc(doc_id, seg), url_oficial=url, pdf_sha1=sha1,
        pagina=h.get("pagina"), parrafo=str(h["parrafo"]) if h.get("parrafo") is not None else None,
        seg=seg, voto_autor=h.get("voto_autor"), caso=h.get("caso"), serie=serie, fecha=h.get("fecha"),
        # El extracto verificado es literal: sus primeras palabras sirven de
        # ancla para que el visor encuentre el pasaje en la página.
        ancla=(" ".join(h["extracto"].split()[:15]) if h.get("extracto") and h.get("verificado") else None),
        cita_canonica=h.get("cita_canonica"), llave=ll, rol_coidh=rol,
        _doc_id=doc_id, _orden=None, _sub=0,
    )


def ficha_supervision(s: Dict[str, Any]) -> Dict[str, Any]:
    """Las supervisiones del 26-nov-2024 que dejaron abiertos arraigo y prisión
    preventiva (temas_mx): no están en la colección, sí en la ficha, con su
    página y un extracto verificado."""
    h = dict(llave=s["doc_id"], doc_id=s["doc_id"], caso=s.get("caso"), fecha=s.get("fecha"),
             seg="considerandos", pagina=s.get("pagina"), extracto=s.get("extracto"),
             verificado=s.get("verificado"), url_oficial=s.get("url_oficial"),
             ubicacion=f"pág. {s.get('pagina')}", cita_canonica=None)
    try:
        import coidh_catalogo
        d = coidh_catalogo.documento(s["doc_id"]) or {}
        h["cita_canonica"] = d.get("cita")
    except Exception:
        pass
    return ficha_hito(h, rol="supervision")


@lru_cache(maxsize=1)
def _mapa_fichas() -> Dict[str, Tuple[str, Any]]:
    """uuid → de qué es ficha. ~2,400 md5 una vez por proceso (≈2 ms)."""
    m: Dict[str, Tuple[str, Any]] = {}
    try:
        import coidh_catalogo
        for doc_id in coidh_catalogo.catalogo()["docs"]:
            m[ficha_id(doc_id)] = ("catalogo", doc_id)
    except Exception:
        pass
    for fid, f in figuras().items():
        for ll, h in _hitos_por_llave(fid).items():
            m[ficha_id(ll)] = ("hito", h)
        for t in (f.get("temas_mx") or {}).values():
            for s in t.get("supervision") or []:
                m[ficha_id(s["doc_id"])] = ("supervision", s)
    return m


def ficha_por_id(uid: str) -> Optional[Dict[str, Any]]:
    """La ficha cuyo id es `uid`, o None. Para /cita: una ficha no vive en
    Qdrant, pero su id se reconstruye (ver `ficha_id`)."""
    x = _mapa_fichas().get(str(uid).lower())
    if not x:
        return None
    clase, obj = x
    if clase == "hito":
        return ficha_hito(obj)
    if clase == "supervision":
        return ficha_supervision(obj)
    return ficha_catalogo(obj)


def publico(d: Dict[str, Any]) -> Dict[str, Any]:
    """Sin las claves internas (las que empiezan por «_»)."""
    return {k: v for k, v in d.items() if not k.startswith("_")}


# ═══════════════════════════════════════════════════════════════ Qdrant

def _expandir(llave: str) -> List[str]:
    """Un voto sin numerar va en ventanas `…|w0`, `…|w1`… (ingesta_coidh.py):
    la llave de la ficha «C-563|v:borea-odria|» las nombra todas."""
    return [f"{llave}w{k}" for k in range(60)] if llave.endswith("|") else [llave]


async def traer_por_llaves(qdrant, llaves: Iterable[str]) -> Dict[str, List[Tuple[str, Dict[str, Any]]]]:
    """llave → [(id, payload)] por `orden`. Un solo scroll, filtro por el campo
    indexado `llave`, sin vectores ni embeddings. LANZA si la colección no está:
    quien llama decide (la línea calla; el resolvedor también)."""
    from qdrant_client.models import FieldCondition, Filter, MatchAny
    pedidas: List[str] = []
    for ll in llaves:
        for x in _expandir(ll):
            if x not in pedidas:
                pedidas.append(x)
    if not pedidas:
        return {}
    pts, _ = await qdrant.scroll(
        collection_name=COLECCION,
        scroll_filter=Filter(must=[FieldCondition(key="llave", match=MatchAny(any=pedidas))]),
        limit=min(512, 6 * len(pedidas)), with_payload=True, with_vectors=False)
    out: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    for p in pts or []:
        pl = p.payload or {}
        ll = str(pl.get("llave") or "")
        out.setdefault(ll, []).append((str(p.id), pl))
        m = re.match(r"^(.*\|)w\d+$", ll)
        if m:                                   # también bajo la llave sin ventana
            out.setdefault(m.group(1), []).append((str(p.id), pl))
    for v in out.values():
        v.sort(key=lambda x: (x[1].get("orden") or 0, x[1].get("sub") or 0))
    return out


async def _scroll(qdrant, must: list, limit: int, order_by=None) -> list:
    from qdrant_client.models import Filter
    kw = dict(collection_name=COLECCION, scroll_filter=Filter(must=must), limit=limit,
              with_payload=True, with_vectors=False)
    if order_by is not None:
        kw["order_by"] = order_by
    pts, _ = await qdrant.scroll(**kw)
    return list(pts or [])


def _parte_con(extracto: Optional[str], partes: List[Tuple[str, Dict[str, Any]]]) -> Tuple[str, Dict[str, Any]]:
    """De un párrafo partido en ventanas, la que contiene el extracto (el
    [Doc ID] debe llevar al texto que se cita), o la primera."""
    if extracto and len(partes) > 1:
        e = _norm(extracto)[:120]
        for x in partes:
            if e and e in _norm(x[1].get("texto_raw") or x[1].get("texto") or ""):
                return x
    return partes[0]


async def traer_citados(qdrant, casos: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Lo que el abogado citó, por llave y con score 1.0 (plan §3.6).

      · con párrafo: el párrafo (todas sus partes) y un vecino a cada lado
        por `orden`;
      · sin párrafo: los resolutivos y los 3 párrafos con más `citado_por_n`
        (sin embeddings: la búsqueda dentro del caso es de F3);
      · resolución que no está en la colección: la ficha del catálogo.
    Lo pedido va primero: la métrica es «primer documento coidh» (revisión A.3).
    LANZA si la colección no existe: sin colección no hay piloto y la consulta
    sigue como antes (tampoco fichas)."""
    from qdrant_client.models import FieldCondition, MatchAny, MatchValue, OrderBy, Range
    ambiguas = any(not c.get("doc_id") and c.get("candidatos") for c in casos)
    casos = [c for c in casos if c.get("doc_id")][:MAX_CASOS]
    if not casos:
        # Sólo citas ambiguas: no hay nada que traer, pero se toca la colección
        # (un punto, sin vectores) para que LANCE si no existe y quien llama
        # no declare la ambigüedad sin piloto (revisión del 25-sep-2026).
        if ambiguas:
            await qdrant.scroll(collection_name=COLECCION, limit=1, with_payload=False, with_vectors=False)
        return []
    llaves = [ll for c in casos for ll in (c.get("llaves") or [])[:MAX_LLAVES_POR_CASO]]
    por_llave = await traer_por_llaves(qdrant, llaves) if llaves else {}

    salida: List[Dict[str, Any]] = []
    vistos: set = set()

    def meter(d: Optional[Dict[str, Any]]):
        if d and d["id"] not in vistos and len(salida) < MAX_UNIDADES_CITADAS:
            vistos.add(d["id"])
            salida.append(d)

    async def hay_doc(doc_id: str) -> bool:
        return bool(await _scroll(qdrant, [FieldCondition(key="doc_id", match=MatchValue(value=doc_id))], 1))

    for c in casos:
        doc_id = c["doc_id"]
        mis_llaves = (c.get("llaves") or [])[:MAX_LLAVES_POR_CASO]
        if mis_llaves:
            hallados = [x for ll in mis_llaves for x in por_llave.get(ll, [])]
            if not hallados:
                ingerido = await hay_doc(doc_id)
                nota = (f"No se encontró {', '.join(ll.split('|', 2)[2] for ll in mis_llaves)} en la resolución "
                        "ingerida: puede no existir ese párrafo. No lo cites como si existiera."
                        if ingerido else None)
                meter(ficha_catalogo(doc_id, c.get("cita_canonica"), c.get("parrafo"), c.get("seg") or "sentencia",
                                     _nombre_autor(c.get("voto_autor")), mis_llaves[0], nota=nota))
                continue
            # Un párrafo partido en ventanas entra entero hasta MAX_PARTES: el
            # ¶47 de la OC-18 son 47 mil tokens y no caben en ningún contexto.
            por_ll: Dict[str, int] = {}
            for pid, pl in hallados:
                ll = str(pl.get("llave") or "")
                por_ll[ll] = por_ll.get(ll, 0) + 1
                if por_ll[ll] <= MAX_PARTES:
                    meter(contrato(pid, pl, rol="pedido"))
            ords = [pl.get("orden") for _, pl in hallados if isinstance(pl.get("orden"), int)]
            if ords:
                vec = sorted({min(ords) - 1, max(ords) + 1} - set(ords))
                vec = [o for o in vec if o >= 0]
                if vec:
                    try:
                        pts = await _scroll(qdrant, [FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
                                                     FieldCondition(key="orden", match=MatchAny(any=vec))], 4)
                        for p in sorted(pts, key=lambda p: (p.payload or {}).get("orden") or 0):
                            meter(contrato(p.id, p.payload or {}, rol="vecino"))
                    except Exception as e:
                        print(f"   ⚖️ COIDH: sin vecinos de {doc_id} ({type(e).__name__})")
            continue

        # Sin párrafo («caso Radilla»): sin embeddings, lo que la propia
        # resolución dice que decidió y lo que más se cita de ella.
        seg = c.get("seg") or "sentencia"
        pts: list = []
        if seg == "voto" and c.get("voto_autor"):
            pts = await _scroll(qdrant, [FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
                                         FieldCondition(key="voto_autor", match=MatchValue(value=c["voto_autor"]))], 64)
            pts = sorted(pts, key=lambda p: (p.payload or {}).get("orden") or 0)[:4]
        if not pts:
            top: list = []
            try:
                top = await _scroll(qdrant, [FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
                                             FieldCondition(key="seg", match=MatchValue(value="sentencia")),
                                             FieldCondition(key="citado_por_n", range=Range(gte=1))], 3,
                                    order_by=OrderBy(key="citado_por_n", direction="desc"))
            except Exception as e:
                print(f"   ⚖️ COIDH: sin los más citados de {doc_id} ({type(e).__name__})")
            res = await _scroll(qdrant, [FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
                                         FieldCondition(key="seg", match=MatchValue(value="resolutivos"))], 40)
            pts = list(top) + sorted(res, key=lambda p: (p.payload or {}).get("orden") or 0)
        if not pts:
            # ¿Ingerida sin resolutivos ni citas (una OC vieja)? Su arranque.
            # ¿No ingerida? La ficha, sin texto.
            pts = await _scroll(qdrant, [FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
                                         FieldCondition(key="orden", match=MatchAny(any=[0, 1, 2]))], 3)
            pts = sorted(pts, key=lambda p: (p.payload or {}).get("orden") or 0)
        if not pts:
            meter(ficha_catalogo(doc_id, c.get("cita_canonica"), None, seg, _nombre_autor(c.get("voto_autor"))))
            continue
        for p in pts:
            pl = p.payload or {}
            meter(contrato(p.id, pl, rol=("resolutivo" if pl.get("seg") == "resolutivos" else "destacado")))
    return salida


async def traer_tesis(qdrant, coleccion: str, registros: Sequence[str]) -> Dict[str, Tuple[str, Dict[str, Any]]]:
    """registro → (id, payload) en `coleccion` (jurisprudencia_nacional_v3).

    Es la ruta que ya existe en la búsqueda directa del chat —filtro exacto
    por el campo `registro`, primero como texto y luego como número—, en una
    sola consulta para todos los registros de la ficha."""
    from qdrant_client.models import FieldCondition, Filter, MatchAny
    regs = [str(r) for r in registros if str(r).isdigit()]
    if not regs:
        return {}
    out: Dict[str, Tuple[str, Dict[str, Any]]] = {}
    for valores in (regs, [int(r) for r in regs]):
        faltan = [v for v in valores if str(v) not in out]
        if not faltan:
            break
        try:
            pts, _ = await qdrant.scroll(
                collection_name=coleccion,
                scroll_filter=Filter(must=[FieldCondition(key="registro", match=MatchAny(any=faltan))]),
                limit=3 * len(faltan), with_payload=True, with_vectors=False)
        except Exception as e:
            print(f"   ⚖️ COIDH: recepción en México no disponible ({type(e).__name__})")
            break
        for p in pts or []:
            reg = str((p.payload or {}).get("registro") or "")
            if reg and reg not in out:
                out[reg] = (str(p.id), p.payload or {})
    return out


async def traer_por_ids(qdrant, coleccion: str, ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    if not ids:
        return {}
    try:
        pts = await qdrant.retrieve(collection_name=coleccion, ids=list(ids), with_payload=True)
    except Exception as e:
        print(f"   ⚖️ COIDH: {coleccion} por id no disponible ({type(e).__name__})")
        return {}
    return {str(p.id): (p.payload or {}) for p in pts or []}


def _doc_doctrina(pid: str, pl: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Un punto de `doctrina` → dict de SearchResult con el contrato del visor
    (el mismo que arma main._sr_doctrina), para que su [Doc ID] resuelva."""
    try:
        import doctrina as _doctrina
        return _doctrina.contrato(_doctrina.fragmento(pid, pl), 0.9)
    except Exception:
        return None


async def traer_cronologia(qdrant, fid: str, ids: Sequence[str], coleccion_tesis: str,
                           norma_a_dict: Optional[Callable[[str, Dict[str, Any], str], Dict[str, Any]]] = None
                           ) -> Dict[str, Dict[str, Any]]:
    """entrada → dict de SearchResult del punto del acervo que la respalda
    (doctrina, Constitución o ley), para que la <entrada> lleve un [Doc ID]
    que el sello reconozca. Lo que no está en el acervo no se inventa: esa
    entrada va sin [Doc ID] y con su URL oficial. Las tesis van por registro
    (traer_tesis), no por aquí."""
    crono = _crono_por_id(fid)
    por_col: Dict[str, List[Tuple[str, str]]] = {}
    for i in ids:
        ea = (crono.get(i) or {}).get("en_acervo") or {}
        if ea.get("coleccion") and ea.get("id") and ea["coleccion"] not in (coleccion_tesis, COLECCION):
            por_col.setdefault(ea["coleccion"], []).append((i, ea["id"]))
    salida: Dict[str, Dict[str, Any]] = {}
    for col, pares in por_col.items():
        pls = await traer_por_ids(qdrant, col, list(dict.fromkeys(pid for _, pid in pares)))
        for i, pid in pares:
            pl = pls.get(pid)
            if pl is None:
                continue
            d = _doc_doctrina(pid, pl) if col == "doctrina" else (norma_a_dict(pid, pl, col) if norma_a_dict else None)
            if d:
                salida[i] = d
    return salida


def medida(xml: str) -> int:
    """Tokens estimados del bloque, sin tiktoken (ver CAR_POR_TOKEN)."""
    return int(len(xml or "") / CAR_POR_TOKEN) + 1


def presupuesto(sel: Dict[str, Any]) -> int:
    """El tope del bloque: PRESUPUESTO_LINEA si se pidió alguna fase; si no,
    PRESUPUESTO_TEMA más PRESUPUESTO_POR_TEMA por cada tema de más, sin pasar
    de PRESUPUESTO_LINEA (ver PRESUPUESTO_POR_TEMA)."""
    if sel.get("fases"):
        return PRESUPUESTO_LINEA
    n = len(sel.get("temas") or [])
    return min(PRESUPUESTO_LINEA, PRESUPUESTO_TEMA + PRESUPUESTO_POR_TEMA * max(0, n - 1))


def _al_presupuesto(sel: Dict[str, Any], hitos: List[Dict[str, Any]], armar: Callable[[], str]) -> Tuple[str, List[str]]:
    """Quita piezas, en el orden de sel["recortables"], hasta que el bloque
    quepa en presupuesto(sel) (26-sep-2026). Toca `sel` y `hitos` en su sitio;
    devuelve (xml, lo recortado). Una tesis que se va se lleva a las parejas
    que sólo ella había traído y a las abandonadas de las que es sustituta:
    una abandonada nunca queda sin su sustituta."""
    xml = armar()
    fuera: List[str] = []
    parejas = sel.get("parejas") or {}
    abandonos = sel.get("abandonos") or {}
    tope = presupuesto(sel)
    for tipo, x in list(sel.get("recortables") or []):
        if medida(xml) <= tope:
            break
        if tipo == "hito" and x in {h["llave"] for h in sel["hitos"]}:
            sel["hitos"] = [h for h in sel["hitos"] if h["llave"] != x]
            hitos[:] = [d for d in hitos if (d.get("_hito") or {}).get("llave") != x]
        elif tipo == "registro" and x in sel["registros"]:
            quitar = {x}
            while True:
                quedan = [r for r in sel["registros"] if r not in quitar]
                huerfanas = {p for p, v in parejas.items() if p in quedan
                             and not any(o in quedan for o in v.get("de") or [])}
                # Y la abandonada cuya sustituta se va, aunque las dos hayan
                # entrado por la lista de un tema y no por `parejas` (14.3,
                # 26-sep-2026): se recortan juntas.
                huerfanas |= {p for p, s in abandonos.items() if p in quedan and s not in quedan}
                if not huerfanas:
                    break
                quitar |= huerfanas
            sel["registros"] = [r for r in sel["registros"] if r not in quitar]
        elif tipo == "cronologia" and x in sel["cronologia"]:
            sel["cronologia"] = [i for i in sel["cronologia"] if i != x]
        elif tipo == "supervision" and sel.get("extracto_supervision"):
            sel["extracto_supervision"] = False
        elif tipo == "tensiones" and sel.get("max_tensiones", 2) > x:
            sel["max_tensiones"] = x
        else:
            continue
        fuera.append(f"{tipo}:{x}")
        xml = armar()
    return xml, fuera


async def traer_linea(qdrant, deteccion: Sequence[Any], coleccion_tesis: str,
                      coleccion_constitucion: str = "bloque_constitucional",
                      tesis_a_dict: Optional[Callable[[str, Dict[str, Any], str], Dict[str, Any]]] = None,
                      norma_a_dict: Optional[Callable[[str, Dict[str, Any], str], Dict[str, Any]]] = None,
                      alcance: Optional[str] = "completa", pregunta: Optional[str] = None,
                      ids: Sequence[str] = ()) -> Optional[Dict[str, Any]]:
    """La línea para (figura, fases): hitos por llave, recepción en México por
    registro, la cronología y los artículos constitucionales que conviven con
    ella.

    `deteccion` es lo que dio pregunta_por_linea —(figura, fases)— o la sonda
    —(figura, fases, ids)—. `alcance` sale de alcance_por_fuentes: con
    «solo_mx» no se toca la colección `coidh` ni se trae nada interamericano;
    con None no se trae nada.

    Devuelve {xml, docs, n, faltan, figura, fases, temas, hitos, registros,
    cronologia, alcance}, o None si no hay nada o si la colección `coidh` no
    responde cuando hace falta. Nunca lanza: una línea que no llega no puede
    costar la consulta.

    `docs` son dicts con los campos de SearchResult: los hitos (silo «coidh»,
    con el contrato) y, si se pasan los conversores de main.py, las tesis, los
    artículos y el acervo de la cronología, para que su [Doc ID] resuelva y el
    sello los vea."""
    try:
        fid, fases = deteccion[0], list(deteccion[1])
        if len(deteccion) > 2 and not ids:
            ids = deteccion[2] or ()
        f = figuras().get(fid)
        if not f or alcance is None:
            return None
        solo_mx = alcance == "solo_mx"
        sel = seleccionar(fid, fases, pregunta=pregunta, alcance=alcance, ids=ids)
        if not sel["hitos"] and not sel["registros"] and not sel["cronologia"]:
            return None
        crono = _crono_por_id(fid)
        hitos: List[Dict[str, Any]] = []
        faltan: List[str] = []
        if sel["hitos"]:
            try:
                puntos = await traer_por_llaves(qdrant, [h["llave"] for h in sel["hitos"]])
            except Exception as e:
                print(f"   ⚖️ COIDH: la colección «{COLECCION}» no responde ({type(e).__name__}): "
                      "la línea no entra y la consulta sigue igual")
                return None
            for h in sel["hitos"]:
                partes = puntos.get(h["llave"])
                if partes:
                    pid, pl = _parte_con(h.get("extracto"), partes)
                    d = contrato(pid, pl, rol="hito")
                    d["ingerido"] = True
                else:
                    d = ficha_hito(h)
                    d["ingerido"] = False
                    faltan.append(h["llave"])
                d["_hito"] = h
                d["_crono"] = crono.get(h["llave"])
                hitos.append(d)
        supervisiones = [] if solo_mx else [dict(ficha_supervision(s), ingerido=False, _sup=s,
                                                 _crono=crono.get(s.get("doc_id")))
                                            for s in sel["supervisiones"]]

        tesis = await traer_tesis(qdrant, coleccion_tesis, sel["registros"])
        normas: Dict[str, Dict[str, Any]] = {}
        ids_norma = {c["qdrant_id"]: c for c in f.get("constitucion_mx") or []
                     if c.get("ref") in sel["constitucion"] and c.get("qdrant_id") and c.get("en_acervo")}
        if ids_norma and not solo_mx:
            normas = await traer_por_ids(qdrant, coleccion_constitucion, list(ids_norma))
        docs_crono = {} if solo_mx else await traer_cronologia(qdrant, fid, sel["cronologia"], coleccion_tesis,
                                                                norma_a_dict)

        sel = dict(sel, hitos=list(sel["hitos"]), registros=list(sel["registros"]), cronologia=list(sel["cronologia"]))
        normas_xml = {pid: ids_norma[pid] for pid in normas}

        def armar() -> str:
            return bloque_xml(fid, sel, hitos, supervisiones, tesis, normas_xml,
                              {i: d.get("id") for i, d in docs_crono.items() if i in sel["cronologia"]})

        xml, recortado = _al_presupuesto(sel, hitos, armar)
        if recortado:
            print(f"   ⚖️ LÍNEA: {len(recortado)} piezas fuera para caber en "
                  f"{presupuesto(sel):,} tokens "
                  f"({', '.join(recortado[:6])}{'…' if len(recortado) > 6 else ''})")
        tesis = {reg: v for reg, v in tesis.items() if reg in sel["registros"]}
        docs_crono = {i: d for i, d in docs_crono.items() if i in sel["cronologia"]}
        faltan = [ll for ll in faltan if ll in {h["llave"] for h in sel["hitos"]}]

        docs = [publico(d) for d in hitos + supervisiones]
        if tesis_a_dict:
            docs += [tesis_a_dict(pid, pl, coleccion_tesis) for pid, pl in tesis.values()]
        if norma_a_dict:
            docs += [norma_a_dict(pid, normas[pid], coleccion_constitucion) for pid in normas]
        ya = {d.get("id") for d in docs}
        for d in docs_crono.values():
            if d.get("id") not in ya:
                docs.append(d)
                ya.add(d.get("id"))
        return dict(xml=xml, docs=docs, n=len(docs), faltan=faltan, figura=fid, fases=sel["fases"],
                    temas=sel["temas"], hitos=[h["llave"] for h in sel["hitos"]], registros=sel["registros"],
                    cronologia=sel["cronologia"], alcance=sel["alcance"], recortado=recortado,
                    presupuesto=presupuesto(sel))
    except Exception as e:
        print(f"   ⚖️ COIDH: la línea falló ({type(e).__name__}: {str(e)[:120]}); la consulta sigue sin ella")
        return None


# ═══════════════════════════════════════════════════════════════ la sonda

# Qué fases abre un tramo cuando la sonda lo encuentra (ver «tramos»).
_FASES_TRAMO = {"t1": ["origen"], "t2": ["evolucion"], "t3": ["evolucion", "mexico"], "t4": ["evolucion", "mexico"],
                "t5": ["mexico", "tema:restricciones_constitucionales"], "t6": ["actual", "mexico"]}
_RX_CANDIDATA = re.compile(
    r"\b(control\w*|constitucion\w*|convencion\w*|tratados?|derechos\s+humanos|corte\s+idh|interamerican\w*|cidh"
    r"|scjn|suprema\s+corte|amparo|jurisprudencia\w*|tesis|restriccion\w*|prision\s+preventiva|arraigo|colegiad\w*"
    r"|supremacia|inaplic\w*|pro\s+persona|reforma\s+(judicial|constitucional))\b")
_DISPONIBLE: Dict[str, Any] = {"t": 0.0, "v": False}
_TTL_DISPONIBLE = 300.0


def umbral() -> float:
    """LINEAS_UMBRAL o UMBRAL_LINEAS (0.55). A calibrar con la colección creada."""
    try:
        return float(os.getenv("LINEAS_UMBRAL") or UMBRAL_LINEAS)
    except ValueError:
        return UMBRAL_LINEAS


def candidata_sonda(texto: str) -> bool:
    """¿Vale la pena preguntarle a la sonda? Sin esto, con la colección creada,
    cada consulta sin detección leería el perfil (la puerta) y pediría un
    embedding: el 99 % de las preguntas no tienen nada que ver con la línea.
    Un filtro de palabras amplio y barato; la decisión la toma el umbral."""
    p = _plegar((texto or "")[:TOPE_PREGUNTA])
    return len(p) >= 20 and bool(_RX_CANDIDATA.search(p))


async def lineas_disponible(qdrant) -> bool:
    """¿Existe la colección `lineas`? Un `count` aproximado, recordado 5 min
    por proceso para no preguntarlo en cada consulta. Hoy: False."""
    ahora = time.monotonic()
    if ahora - _DISPONIBLE["t"] < _TTL_DISPONIBLE:
        return bool(_DISPONIBLE["v"])
    try:
        await qdrant.count(collection_name=LINEAS, exact=False)
        v = True
    except Exception:
        v = False
    _DISPONIBLE.update(t=ahora, v=v)
    return v


async def sondear(qdrant, vector: Optional[Sequence[float]], umbral_min: Optional[float] = None,
                  limite: int = 8) -> Optional[Tuple[str, List[str], List[str]]]:
    """La sonda semántica: la pregunta (su embedding de 1536, el mismo
    text-embedding-3-small de la búsqueda) contra la colección `lineas`, que
    tiene un punto por entrada de la cronología, por hito, por tramo y un mapa
    (scripts/lineas_qdrant.py). Para lo que las palabras no alcanzan: «¿los
    jueces pueden dejar de aplicar la Constitución si choca con un tratado?»
    no dice «convencionalidad» ni «restricción».

    Devuelve (figura, fases, ids) o None: sin vector, sin colección, si falla o si nada pasa del umbral. `ids` son las entradas,
    llaves o registros que pasaron, en orden de score; `fases` salen de los
    tramos de lo encontrado (_FASES_TRAMO). Nunca lanza."""
    if not vector:
        return None
    u = umbral() if umbral_min is None else umbral_min
    try:
        res = await qdrant.query_points(collection_name=LINEAS, query=list(vector), using="dense", limit=limite,
                                        with_payload=True, score_threshold=u)
        pts = list(getattr(res, "points", None) or [])
    except Exception as e:
        print(f"   ⚖️ LINEAS: la sonda no responde ({type(e).__name__}); se sigue sin ella")
        return None
    pts = [p for p in pts if (getattr(p, "score", 0.0) or 0.0) >= u and (p.payload or {}).get("linea") in figuras()]
    if not pts:
        return None
    figura = Counter((p.payload or {}).get("linea") for p in pts).most_common(1)[0][0]
    fases: List[str] = []
    ids: List[str] = []
    for p in pts:
        pl = p.payload or {}
        if pl.get("linea") != figura:
            continue
        if pl.get("tipo_ficha") == "mapa":
            fases += ["evolucion", "mexico"]
        elif pl.get("tramo"):
            fases += _FASES_TRAMO.get(pl["tramo"], [])
        if pl.get("tipo_ficha") == "hito":
            ids += [str(x) for x in (pl.get("id"), pl.get("llave"), pl.get("registro")) if x]
    return figura, list(dict.fromkeys(fases)) or ["concepto"], list(dict.fromkeys(ids))


# ═══════════════════════════════════════════════════════════════ los bloques

_INSTRUCCION_CITA = (
    "CÓMO SE CITA LA CORTE IDH (25-sep-2026):\n"
    "· Sentencia: «Corte IDH. Caso X Vs. Y. [Excepciones…, Fondo…]. Sentencia de [fecha]. "
    "Serie C No. N, párr. P [Doc ID: uuid]». La cita exacta viene en <cita>: cópiala y añade el [Doc ID]; "
    "si son varios párrafos, un [Doc ID] por párrafo, cada uno en sus corchetes (nunca «Doc IDs»).\n"
    "· Voto: «voto razonado del juez X en el Caso …, párr. P [Doc ID: uuid]». UN VOTO NO ES LA CORTE: "
    "nunca le atribuyas a la Corte lo que dice un voto, ni al revés.\n"
    "· Opinión consultiva: «Corte IDH. Opinión Consultiva OC-N/AA …, párr. P [Doc ID: uuid]». "
    "Resolutivos: «punto resolutivo N». Supervisión: «considerando N».\n"
    "· Di «en el sistema interamericano»; NUNCA «en toda la historia» ni «por primera vez en la historia».\n"
    "· Esto va ADEMÁS de la Constitución, la ley y la jurisprudencia mexicanas de <documentos>, "
    "NUNCA en su lugar.\n"
    "· Cita textual sólo lo que está aquí, entre comillas « » y carácter por carácter; si no, parafrasea "
    "sin comillas y cita igual el [Doc ID]. Lo que no está aquí no se transcribe ni se inventa.")


def _g(x: Any, k: str) -> Any:
    return x.get(k) if isinstance(x, dict) else getattr(x, k, None)


def _tipo_xml(x: Any) -> str:
    return str(_g(x, "tipo") or "sentencia_coidh").upper()


def bloque_resueltos_xml(docs: Sequence[Any], ambiguos: Sequence[Dict[str, Any]] = ()) -> str:
    """<casos_corte_idh>: los párrafos que el abogado citó, en su propio bloque.

    Va DESPUÉS de <documentos> y no pasa por reorder_by_hierarchy (revisión
    A.3): con `"coidh": 3` el párrafo pedido quedaba detrás de toda la
    Constitución y de todas las leyes. Aquí va primero lo pedido, luego sus
    vecinos (marcados), luego las fichas sin texto; y las citas ambiguas se
    declaran para que el modelo PREGUNTE en vez de elegir."""
    docs = [d for d in docs if _g(d, "silo") == SILO]
    if not docs and not ambiguos:
        return ""
    rango = {"pedido": 0, "resolutivo": 1, "destacado": 1, "vecino": 2, "hito": 3, "supervision": 3, "ficha": 4}
    docs = sorted(docs, key=lambda d: rango.get(_g(d, "rol_coidh") or "pedido", 2))
    partes = ["<casos_corte_idh>",
              "<!-- INSTRUCCIÓN CASOS CORTE IDH: el abogado citó estos casos y aquí está su texto oficial, "
              "párrafo por párrafo, del PDF de corteidh.or.cr. Responde con ellos PRIMERO, en el orden en que "
              "los pidió. rol=\"vecino\" es el párrafo anterior o siguiente del pedido: úsalo sólo para "
              "entender el contexto. rol=\"ficha\" NO tiene texto ingerido: cita el caso con su <cita> y su "
              "[Doc ID], y di que el texto del párrafo no está en el acervo; no lo transcribas ni lo "
              "parafrasees de memoria.\n" + _INSTRUCCION_CITA + " -->"]
    for d in docs:
        attrs = [f'id="{_esc(_g(d, "id"))}"', f'tipo="{_esc(_tipo_xml(d))}"',
                 'jerarquia="JURISPRUDENCIA_INTERAMERICANA"', f'rol="{_esc(_g(d, "rol_coidh") or "pedido")}"']
        for k, nombre in (("caso", "caso"), ("serie", "serie"), ("fecha", "fecha"), ("parrafo", "parrafo"),
                          ("pagina", "pagina"), ("seg", "seg"), ("voto_autor", "voto")):
            v = _g(d, k)
            if v not in (None, ""):
                attrs.append(f'{nombre}="{_esc(v)}"')
        texto = str(_g(d, "texto") or "")
        if len(texto) > 6000:
            texto = texto[:6000] + "... [truncado]"
        partes.append(f"<documento {' '.join(attrs)}>\n<cita>{_esc(_g(d, 'cita_canonica'))} "
                      f"[Doc ID: {_esc(_g(d, 'id'))}]</cita>\n{_esc(texto)}\n</documento>")
    for a in ambiguos:
        cands = "; ".join(f"{c.get('caso')} ({c.get('doc_id')}, {c.get('fecha') or 's/f'})"
                          for c in (a.get("candidatos") or [])[:8])
        partes.append(f'<cita_ambigua evidencia="{_esc(a.get("evidencia"))}">La cita puede ser de varios '
                      f"casos: {_esc(cands)}. No elijas: pregunta al abogado a cuál se refiere.</cita_ambigua>")
    partes.append("</casos_corte_idh>")
    return "\n".join(partes)


def _hito_xml(d: Dict[str, Any]) -> str:
    """Un hito, en lo justo (~240 tokens): el caso, la serie y el tipo ya van
    en la <cita>, y el id una sola vez, en su [Doc ID]. Desde el pilar
    (25-sep-2026) lleva también su postura y su fuerza —una sentencia ORDENA,
    un voto PROPONE—, y verificacion="parcial" si su extracto sólo se cotejó en
    una copia no oficial (entonces no se imprime)."""
    h = d.get("_hito") or {}
    c = d.get("_crono") or {}
    attrs = [f'fecha="{_esc(d.get("fecha"))}"', f'fase="{_esc(h.get("fase"))}"', f'arista="{_esc(h.get("arista"))}"']
    for k, nombre in (("parrafo", "parrafo"), ("pagina", "pagina"), ("voto_autor", "voto")):
        if d.get(k) not in (None, ""):
            attrs.append(f'{nombre}="{_esc(d[k])}"')
    attrs.append(f'ingerido="{"sí" if d.get("ingerido") else "no"}"')
    # Sólo lo que dice algo: una sentencia que no toca la restricción
    # constitucional (postura «no_aplica») y obliga (lo normal) no lo repite.
    if c.get("postura") not in (None, "no_aplica"):
        attrs.append(f'postura="{_esc(c["postura"])}"' + (f' sentido="{_esc(c["sentido"])}"' if c.get("sentido") else ""))
    if c.get("fuerza") not in (None, "obligatoria"):
        attrs.append(f'fuerza="{_esc(c["fuerza"])}"')
    if c.get("vigencia") not in (None, "vigente", "no_aplica"):
        attrs.append(f'vigencia="{_esc(c["vigencia"])}"')
    if not h.get("verificado", True):
        attrs.append('verificacion="parcial"')
    cuerpo = [f"<cita>{_esc(d.get('cita_canonica'))} [Doc ID: {_esc(d['id'])}]</cita>"]
    if h.get("ubicacion") and not d.get("parrafo"):
        cuerpo.append(f"<ubicacion>{_esc(h['ubicacion'])}</ubicacion>")
    if h.get("aporte"):
        cuerpo.append(f"<aporte>{_esc(h['aporte'])}</aporte>")
    if h.get("extracto") and h.get("verificado", True):
        cuerpo.append(f"<extracto>«{_esc(h['extracto'])}»</extracto>")
    # Las notas largas son de verificación (qué PDF, qué página se corrigió):
    # al modelo le sirven las cortas y las del núcleo («no revisé toda la
    # jurisprudencia anterior», «sólo dice convencionalidad»).
    if h.get("nota") and (len(h["nota"]) <= 140 or h.get("llave") in NUCLEO):
        cuerpo.append(f"<nota>{_esc(h['nota'])}</nota>")
    return f"<hito {' '.join(attrs)}>\n" + "\n".join(cuerpo) + "\n</hito>"


def _precision_historica(hitos: Sequence[Dict[str, Any]]) -> str:
    """Expresión (García Ramírez, 2003) → Corte en pleno (Almonacid ¶124) →
    ex officio (Cesados ¶128). Con fechas, párrafo, página y [Doc ID]; lo que
    no se trajo no se cita con id."""
    por = {d.get("llave"): d for d in hitos}

    def ref(ll: str) -> str:
        d = por.get(ll)
        if not d:
            return ""
        return (f"{d.get('caso')}, {fecha_larga(d.get('fecha'))}, párr. {d.get('parrafo')}, "
                f"pág. {d.get('pagina')} [Doc ID: {d['id']}]")
    a, b, c = (ref(ll) for ll in NUCLEO)
    if not (a or b or c):
        return ""
    txt = ["<precision_historica>",
           "En el sistema interamericano (no «en toda la historia»):"]
    if a:
        txt.append(f"1. La expresión «control de convencionalidad» aparece primero en votos del juez Sergio "
                   f"García Ramírez: voto concurrente razonado en {a}. Ahí el control lo ejerce la propia "
                   "Corte IDH, no los jueces nacionales; es un voto, no la Corte.")
    if b:
        txt.append(f"2. La Corte en pleno lo formula por primera vez en {b}: el Poder Judicial interno debe "
                   "ejercer «una especie de» control de convencionalidad. Todavía no dice «ex officio». "
                   "Ese párrafo es de Almonacid (Serie C No. 154), NO de Trabajadores Cesados.")
    if c:
        txt.append(f"3. El «ex officio» llega dos meses después, en {c}: desaparece «una especie de» y el "
                   "control se ejerce de oficio, en el marco de las competencias y regulaciones procesales.")
    txt.append("</precision_historica>")
    return "\n".join(txt)


_INSTRUCCION_PILAR = (
    "Cuéntala en orden cronológico uniendo <hitos>, <recepcion_mx> y <cronologia> por su fecha. postura=, "
    "fuerza= y vigencia= dicen quién ORDENA, RECOMIENDA o PROPONE y si sigue vigente (vigencia=\"abandonada\": "
    "NO es criterio vigente, cita su reemplazo); sigue las <reglas> y fecha el estado actual con <cortes>.")

_INSTRUCCION_SOLO_MX = (
    "<!-- INSTRUCCIÓN LÍNEA JURISPRUDENCIAL (sólo México): el abogado apagó «constitucional» en «Fuentes» y dejó "
    "«jurisprudencia»: aquí van SÓLO tesis y resoluciones mexicanas (SCJN, Plenos Regionales, Tribunales "
    "Colegiados). No cites a la Corte IDH, a la CIDH, a la ONU ni tratados de memoria. Cuéntala en orden "
    "cronológico por su fecha. Una tesis con vigencia=\"abandonada\" o \"superada\" NO es criterio vigente: dilo y "
    "cita la que la reemplaza. Cita cada tesis con su clave, su registro y su propio [Doc ID] (uno por "
    "corchetes; nunca «Doc IDs»). Fecha el estado actual con "
    "<cortes> («según lo verificado hasta…»), nunca «hoy». -->")


def _palabras(s: Any, n: int) -> str:
    w = str(s or "").split()
    return " ".join(w[:n]) + ("…" if len(w) > n else "")


# Abreviaturas que terminan en punto y NO cierran la frase: «art.», «fr.»,
# «párr.», «P./J.», «1o.», «Vs.»… Cortar ahí mutila igual que cortar a media
# palabra.
_RX_ABREVIATURA = re.compile(
    r"^[(«\"“]?(?:(?i:arts?|frs?|p[aá]rrs?|p[aá]gs?|pp?|n[uú]ms?|no|vs|cons|inc|cfr|lit|vol|t|ed|ob|cit|op|ss|lic"
    r"|dr|dra|mtro|mtra)\.|[A-ZÁÉÍÓÚÑ]\.|\d+[oa]\.|\w{1,3}\.(?:/?\w{1,3}\.)+)$")


def _recorte(s: Any, n: int) -> str:
    """Hasta n palabras, cortando en el último fin de frase («.», «;» o «:»)
    de la segunda mitad; si no hay, a n palabras con «…».

    POR QUÉ (revisión adversarial, 26-sep-2026): el recorte a 32 palabras
    dejaba el aporte de la AI 130/2019 en «…la interpretación conforme del
    art. 19 no tuvo…» y se perdía «mayoría y la P./J. 20/2014 quedó intacta»,
    que es justo lo que decide si esa postura prosperó. Una frase entera y
    más corta dice más que una larga cortada."""
    w = str(s or "").split()
    if len(w) <= n:
        return " ".join(w)
    for i in range(n, max(1, n // 2), -1):
        p = w[i - 1].rstrip("»”\")")
        if p.endswith((";", ":")) or (p.endswith(".") and not _RX_ABREVIATURA.search(p)):
            return " ".join(w[:i])
    return " ".join(w[:n]) + "…"


def _reglas_para(f: Dict[str, Any], sel: Dict[str, Any], vigencias: Iterable[str], clases: Iterable[str]) -> List[str]:
    """Las reglas de redacción que tocan a lo que entró, no las doce siempre
    (cada una cuesta ~30 tokens). Se eligen por lo que dicen, no por su lugar
    en la lista: si mañana se reordenan, no se cruzan."""
    reglas = f.get("reglas_de_redaccion") or []
    temas = set(sel.get("temas") or [])
    solo_mx = sel.get("alcance") == "solo_mx"
    vig = set(vigencias)
    cls = set(clases)
    ppo = bool(temas & {"prision_preventiva_oficiosa", "arraigo"}) or sel.get("actual_mx")
    restr = bool(temas & {"restricciones_constitucionales", "amparo_reformas_constitucionales"}) or sel.get("actual_mx")
    salida = []
    for r in reglas:
        p = _plegar(r)
        if "sistema interamericano" in p or "un voto no es la corte" in p:
            ok = False            # ya lo dice _INSTRUCCION_CITA, que va en todo bloque con la Corte IDH
        elif "ordena" in p and "recomienda" in p:
            ok = not solo_mx
        elif "proyecto no votado" in p:
            ok = "scjn_proyecto" in cls
        elif "abandonada o superada" in p:
            ok = bool(vig & {"abandonada", "superada", "superada_en_parte"})
        elif "p./j. 64/2014" in p:
            ok = restr or ppo
        elif "literalidad" in p or "166" in p or "arraigo y la prision" in p:
            ok = ppo
        elif "12a. epoca" in p:
            ok = bool(cls & {"sala", "pleno_regional"})
        elif "dos posturas" in p:
            ok = restr and not solo_mx
        else:
            ok = True
        if ok:
            salida.append(r)
    return salida


def _entrada_xml(e: Dict[str, Any], doc_id: Optional[str]) -> str:
    """Una entrada de la cronología (~110 tokens): quién, cuándo, qué fuerza y
    qué aporta, con su extracto literal (recortado) y su [Doc ID] si está en
    el acervo; si no, la URL oficial, sin inventar un id."""
    fecha = e.get("fecha_resolucion") or e.get("fecha_publicacion")
    attrs = [f'fecha="{_esc(fecha)}"', f'organo="{_esc(_palabras(e.get("organo"), 6))}"', f'tipo="{_esc(e.get("tipo"))}"']
    for k in ("clave", "registro"):
        if e.get(k):
            attrs.append(f'{k}="{_esc(e[k])}"')
    for k in ("vigencia", "reemplazo", "postura", "sentido", "fuerza"):
        if e.get(k) and e[k] != "no_aplica":
            attrs.append(f'{k}="{_esc(e[k])}"')
    # Lo que no se cotejó entero lo dice, como el hito (26-sep-2026: la CC
    # 3/2026 del Pleno Regional sólo consta en prensa porque la SCJN pidió
    # suspender su publicación).
    if (e.get("verificacion") or {}).get("estado") == "parcial":
        attrs.append('verificacion="parcial"')
    cuerpo = [_esc(_palabras(e.get("titulo"), 8)) + "."]
    if e.get("aporte"):
        cuerpo.append(_esc(_recorte(e["aporte"], MAX_PALABRAS_APORTE)))
    if e.get("extracto"):
        cuerpo.append(f"«{_esc(_palabras(e['extracto'], 16))}»")
    if e.get("vigencia") not in ("vigente", "no_aplica") and e.get("vigencia_nota"):
        cuerpo.append(f"({_esc(_recorte(e['vigencia_nota'], MAX_PALABRAS_APORTE))})")
    if doc_id:
        cuerpo.append(f"[Doc ID: {_esc(doc_id)}]")
    elif e.get("fuente_tipo") == "prensa":
        cuerpo.append(f"(sólo consta en prensa; no hay versión oficial publicada: {_esc(e.get('url_oficial'))})")
    else:
        cuerpo.append(f"(no está en el acervo: {_esc(e.get('url_oficial'))})")
    return f"<entrada {' '.join(attrs)}>" + " ".join(cuerpo) + "</entrada>"


def _tesis_xml(pid: str, pl: Dict[str, Any], r: Dict[str, Any], e: Optional[Dict[str, Any]], reg: str) -> str:
    """Una tesis de la recepción con su vigencia a la vista (25-sep-2026: el
    modelo recibió la P. X/2015 como «TESIS AISLADA» del Pleno sin ninguna
    advertencia y la dio por vigente)."""
    clave = pl.get("clave_tesis") or pl.get("numero_tesis") or pl.get("tesis") or r.get("clave") or ""
    rubro = (pl.get("rubro") or r.get("rubro_abreviado") or "").strip()
    vig = r.get("vigencia") or "vigente"
    # Sin atributo id: el mismo uuid va en el [Doc ID] (eran ~25 tokens por tesis).
    attrs = [f'registro="{_esc(reg)}"', f'clave="{_esc(clave)}"',
             f'instancia="{_esc(pl.get("instancia") or r.get("instancia"))}"',
             f'fecha="{_esc(r.get("fecha_publicacion") or pl.get("fecha_publicacion"))}"',
             f'fuerza="{_esc(r.get("fuerza"))}"', f'vigencia="{_esc(vig)}"']
    if r.get("reemplazo"):
        attrs.append(f'reemplazo="{_esc(r["reemplazo"])}"')
    if r.get("sustituye"):
        attrs.append(f'sustituye="{_esc(",".join(r["sustituye"]))}"')
    if r.get("postura") not in (None, "no_aplica"):
        attrs.append(f'postura="{_esc(r["postura"])}"')
    aviso = ""
    if vig == "abandonada":
        aviso = f"ABANDONADA por la tesis de registro {r.get('reemplazo')}: NO la presentes como vigente. "
    elif vig in ("superada", "superada_en_parte"):
        aviso = ("SUPERADA" + (" EN PARTE" if vig == "superada_en_parte" else "")
                 + (f" por la tesis de registro {r.get('reemplazo')}" if r.get("reemplazo") else "") + ". ")
    if vig in ("abandonada", "superada"):
        # La abandonada o superada va corta: su rubro para reconocerla y el
        # aviso; lo que hay que citar es su reemplazo, que va entero.
        return (f"<tesis {' '.join(attrs)}>{_esc(aviso)}Decía: {_esc(_palabras(rubro, 18))}"
                f" [Doc ID: {_esc(pid)}]</tesis>")
    porque = (e or {}).get("aporte") or r.get("porque")
    nota = r.get("vigencia_nota") if (vig != "vigente" or re.search(
        r"pendiente|sub iudice|desplaz|inaplicable|disputad", str(r.get("vigencia_nota") or ""), re.I)) else None
    # La que sustituye a otra va con su aporte entero: es la que hay que citar
    # (el de la P./J. 2/2022 cortado a 26 palabras perdía «y sobre las normas
    # aplicadas en el acto reclamado», que es la respuesta a David).
    return (f"<tesis {' '.join(attrs)}>{_esc(aviso)}{_esc(_palabras(rubro, 22))}"
            + (f" — {_esc(_palabras(porque, 50 if r.get('sustituye') else 26))}" if porque else "")
            # La nota entera hasta su último fin de frase (26-sep-2026: «disputada:
            # un colegiado la tiene por inaplicable; el Pleno Regional…» cortada a
            # 16 palabras perdía la mitad que dice quién sostiene lo contrario).
            + (f" ({_esc(_recorte(nota, MAX_PALABRAS_APORTE))})" if nota else "")
            + f" [Doc ID: {_esc(pid)}]</tesis>")


def _cortes_xml(f: Dict[str, Any]) -> str:
    c = f.get("cortes") or {}
    if not c:
        return ""
    trozos = []
    if c.get("corte_idh"):
        trozos.append(f"Corte IDH hasta {c['corte_idh'].get('fecha')} (México completo; de otros países faltan "
                      "sentencias de 2024-2026)")
    if c.get("acervo_sjf"):
        trozos.append(f"acervo de tesis hasta {c['acervo_sjf'].get('fecha')}")
    if c.get("api_sjf"):
        trozos.append(f"Semanario hasta el registro {c['api_sjf'].get('registro')} ({c['api_sjf'].get('fecha')}; "
                      "después, laguna)")
    if c.get("dof_y_acuerdos"):
        trozos.append(f"DOF y acuerdos hasta {c['dof_y_acuerdos'].get('fecha')}")
    return "<cortes>" + _esc("; ".join(trozos)) + ".</cortes>"


def bloque_xml(fid: str, sel: Dict[str, Any], hitos: Sequence[Dict[str, Any]],
               supervisiones: Sequence[Dict[str, Any]] = (),
               tesis: Optional[Dict[str, Tuple[str, Dict[str, Any]]]] = None,
               normas: Optional[Dict[str, Dict[str, Any]]] = None,
               docs_crono: Optional[Dict[str, Optional[str]]] = None) -> str:
    """<linea_jurisprudencial>: los hitos en orden CRONOLÓGICO, cada uno con
    caso, fecha, párrafo, página, aporte, extracto y [Doc ID]; la precisión
    histórica del origen; tensiones; recepción en México (registros de la
    SCJN) con su vigencia; la <cronologia> del pilar; y «según lo ingerido
    hasta [fecha]» con los <cortes> de cada fuente. Se añade después del
    contexto normal (patrón de la doctrina): la jerarquía rompería la
    cronología. Con alcance «solo_mx», sólo lo mexicano."""
    f = figuras().get(fid) or {}
    tesis = tesis or {}
    normas = normas or {}
    docs_crono = docs_crono or {}
    crono = _crono_por_id(fid)
    vig = f.get("vigente_al") or lineas().get("generado") or ""
    temas = sel.get("temas") or []
    reales = sel.get("fases") or []
    solo_mx = sel.get("alcance") == "solo_mx"
    alcance = ("línea completa" if reales and not sel.get("tema_primero") else "hitos del tema, no la línea entera")
    fichas = {str(r.get("registro")): r for r in f.get("recepcion_mx") or []}
    entradas = [crono[i] for i in sel.get("cronologia") or [] if i in crono]
    regs_xml = [reg for reg in sel.get("registros") or [] if reg in tesis]
    vigencias = [(fichas.get(r) or {}).get("vigencia") or "vigente" for r in regs_xml] + [e.get("vigencia") for e in entradas]
    clases = [e.get("tipo") for e in entradas] + [
        {"Primera Sala": "sala", "Segunda Sala": "sala", "Plenos Regionales": "pleno_regional"}.get(
            (fichas.get(r) or {}).get("instancia"), "") for r in regs_xml]

    # La fecha de ingesta de la Corte IDH NO es la del bloque (revisión
    # adversarial, 26-sep-2026): en solo_mx no hay nada interamericano y aun
    # así abría con vigente_al="2025-09-30" y cerraba «según lo ingerido hasta
    # 2025-09-30», con tesis de 2026 dentro; el modelo podía fechar así la
    # postura de la SCJN. Lo mexicano se fecha con <cortes>.
    partes = [f'<linea_jurisprudencial figura="{_esc(f.get("nombre") or fid)}" '
              f'fases="{_esc(",".join(reales))}" temas="{_esc(",".join(temas))}" '
              f'alcance="{_esc(alcance)}" curada="sí"'
              + (' modo="solo_mx"' if solo_mx else f' corte_idh="{_esc(vig)}"') + ">"]
    # Tensiones: con la línea, salvo que sólo se pregunte el origen; con un
    # tema, las que tocan sus hitos (la de prisión preventiva, la del arraigo).
    # Se calculan ANTES de la instrucción, que sólo las pide si entraron.
    ten = [] if solo_mx or reales == ["origen"] else (f.get("tensiones") or [])
    # Las que tocan lo que entró (hitos o tesis), como mucho tres: con la
    # línea entera eran seis y ~900 tokens, las de México también en una
    # pregunta sin México (25-sep-2026).
    mias = {d.get("llave") for d in hitos} | set(regs_xml)
    ten = [t for t in ten if mias & set((t.get("a_favor") or []) + (t.get("en_contra") or [])
                                        + [str(x) for x in t.get("registros") or []])]
    # PRIMERO LAS DE LOS TEMAS PEDIDOS (reverificación, 26-sep-2026). Iban en
    # el orden del archivo y el tope de dos se lo quedaban «opiniones
    # consultivas como parámetro» y «res interpretata» —tocan la P./J. 2/2025 y
    # la P./J. 21/2014 del núcleo mexicano—; medido con Qdrant real, en «Traza
    # la línea… qué posturas serias hay para inaplicar restricciones
    # constitucionales como la prisión preventiva oficiosa» y en «¿Qué posturas
    # serias hay…?» no entraba ni «restricciones constitucionales frente a la
    # Convención» ni «¿puede un juez inaplicar la prisión preventiva
    # oficiosa?», que son justo las posturas que se preguntan. Primero las que
    # comparten un hito con un tema pedido, luego las que comparten una tesis;
    # en empate, el orden del archivo (sin temas, el de siempre).
    temas_mx = f.get("temas_mx") or {}
    ll_tema = {ll for t in temas for ll in (temas_mx.get(t) or {}).get("hitos") or []}
    reg_tema = {str(r) for t in temas for r in (temas_mx.get(t) or {}).get("registros") or []}

    def _rango_tension(t: Dict[str, Any]) -> int:
        if ll_tema & set((t.get("a_favor") or []) + (t.get("en_contra") or [])):
            return 0
        return 1 if reg_tema & {str(x) for x in t.get("registros") or []} else 2
    ten = sorted(ten, key=_rango_tension)[:sel.get("max_tensiones", 2)]

    # La instrucción nombra SÓLO las partes que entraron. Primero fueron los
    # ejes (quién está obligado, parámetro, efectos, forma): si no entró
    # ninguno, el modelo los contaba de memoria, sin [Doc ID] (revisión
    # adversarial, 26-sep-2026). Después, en la reverificación del mismo día
    # (hallazgo 12b): en «Traza la línea…» el presupuesto quitaba las dos
    # <tensiones> y la instrucción seguía pidiendo «→ estado actual →
    # tensiones → recepción en México»; «¿Dónde nace…?» no trae ni estado
    # actual ni tensiones y también las pedía. Cada tramo de la columna va
    # sólo si su pieza está en el bloque.
    fases_hitos = {((d.get("_hito") or {}).get("fase") or "").lower() for d in hitos}
    hay_nucleo = any(d.get("llave") in NUCLEO for d in hitos)
    hay_ejes = bool(fases_hitos & _FASES_EJE)
    hay_actual = bool(fases_hitos & _FASES_HITO["actual"]) or bool(sel.get("actual_mx"))
    columna = ((["expresión (votos de García Ramírez; el control lo ejerce la Corte IDH)",
                 "formulación en pleno (Almonacid ¶124)", "ex officio (Cesados ¶128)"] if hay_nucleo else [])
               + (["ejes (quién está obligado, parámetro, efectos, forma)"] if hay_ejes else [])
               + (["estado actual"] if hay_actual else [])
               + (["tensiones"] if ten else [])
               + (["recepción en México"] if regs_xml else []))
    if solo_mx:
        partes.append(_INSTRUCCION_SOLO_MX)
    else:
        partes.append(
            "<!-- INSTRUCCIÓN LÍNEA JURISPRUDENCIAL (Corte IDH): "
            + ("esta es la línea curada y verificada contra el PDF oficial de cada resolución. "
               + (f"Su columna: {' → '.join(columna)}. " if columna else "")
               if alcance == "línea completa" else
               "éstos son SÓLO los hitos y criterios que tocan el tema de la pregunta, verificados; no cuentes "
               "la línea entera. Preséntalos después de la norma y la jurisprudencia mexicanas"
               + (", con sus tensiones. " if ten else ". "))
            + _INSTRUCCION_PILAR
            + f" Del estado actual de la Corte IDH di «según lo ingerido hasta {vig}», nunca «hoy». ingerido=\"no\": "
            "el texto completo no está en Iurexia, sólo su extracto verificado; cita igual su [Doc ID].\n"
            + _INSTRUCCION_CITA + " -->")
    reglas = _reglas_para(f, sel, vigencias, clases)
    if reglas:
        partes.append("<reglas>" + " ".join(f"{i}. {_esc(r)}" for i, r in enumerate(reglas, 1)) + "</reglas>")
    if reales and not solo_mx and not sel.get("tema_primero"):
        pre = _precision_historica(hitos)
        if pre:
            partes.append(pre)
        # El resumen del origen ya lo dice la precisión histórica.
        resumen = f.get("resumen") or {}
        claves = []
        for x in reales:
            claves += {"evolucion": ["evolucion"], "actual": ["estado_actual"],
                       "concepto": ["estado_actual"]}.get(x, [])
        if sel.get("mexico"):
            # Con México basta su resumen, que ya trae el estado a sep-2026:
            # los tres juntos eran ~1,000 tokens (medido el 25-sep-2026).
            claves = ["mexico"]
        for k in dict.fromkeys(claves):
            if resumen.get(k):
                partes.append(f'<resumen parte="{_esc(k)}">{_esc(resumen[k])}</resumen>')
    elif temas and sel.get("tema_primero") and not solo_mx:
        # Si además de un tema se pide su fase («¿cuál es la postura actual
        # sobre…?»), el resumen del tramo del pilar: el hilo que une sus
        # piezas. Sin fase, las piezas y la advertencia bastan (el de la
        # prisión preventiva son ~1,300 tokens).
        tid = next((_TRAMO_TEMA[t] for t in temas if t in _TRAMO_TEMA), None)
        t = (f.get("tramos") or {}).get(tid) if tid else None
        if t:
            partes.append(f'<resumen parte="tramo:{_esc(tid)}" titulo="{_esc(t.get("titulo"))}" '
                          f'periodo="{_esc(t.get("periodo"))}">{_esc(t.get("resumen"))}</resumen>')
    for adv in sel.get("advertencias") or []:
        if not solo_mx or not re.search(r"Corte IDH|supervisi", adv):
            partes.append(f"<advertencia>{_esc(adv)}</advertencia>")

    if hitos:
        partes.append('<hitos orden="cronologico">')
        partes += [_hito_xml(d) for d in hitos]
        partes.append("</hitos>")

    # El extracto de la supervisión sólo con su tema (prisión preventiva,
    # arraigo): en la línea entera basta su estado, «abierto».
    con_extracto = sel.get("extracto_supervision", bool(set(temas) & {"prision_preventiva_oficiosa", "arraigo"}))
    for s in supervisiones:
        sup = s.get("_sup") or {}
        if not con_extracto:
            partes.append(f'<supervision fecha="{_esc(sup.get("fecha"))}" caso="{_esc(sup.get("caso"))}" '
                          f'pagina="{_esc(sup.get("pagina"))}" ingerido="no">{_esc(sup.get("estado"))} '
                          f"[Doc ID: {_esc(s['id'])}]</supervision>")
            continue
        partes.append(f'<supervision id="{_esc(s["id"])}" fecha="{_esc(sup.get("fecha"))}" '
                      f'caso="{_esc(sup.get("caso"))}" pagina="{_esc(sup.get("pagina"))}" ingerido="no" '
                      f'postura="ordena_adecuar">\n'
                      f"<estado>{_esc(sup.get('estado'))}</estado>\n"
                      + (f"<extracto>«{_esc(sup.get('extracto'))}»</extracto>\n"
                         if sup.get("verificado") and con_extracto else "")
                      + f"<cita>{_esc(s.get('cita_canonica'))} [Doc ID: {_esc(s['id'])}]</cita>\n</supervision>")

    # Las tensiones ya se eligieron arriba (antes de la instrucción).
    if ten:
        partes.append("<tensiones>")
        for t in ten:
            regs = ", ".join(t.get("registros") or [])
            partes.append(f'<tension tema="{_esc(t.get("tema"))}">{_esc(t.get("resumen"))}'
                          + (f" (registros: {_esc(regs)})" if regs else "") + "</tension>")
        partes.append("</tensiones>")

    if regs_xml:
        partes.append("<recepcion_mx>")
        partes.append("<!-- Criterios de la SCJN y de tribunales mexicanos: son derecho mexicano y se citan "
                      "como cualquier tesis, con su registro y su [Doc ID]. Mira vigencia= antes de citarla. -->")
        # En orden de fecha, como <hitos> y <cronologia> (revisión adversarial,
        # 26-sep-2026: salía en orden de prioridad —2014, 2014, 2022, 2014,
        # 2025…— y la instrucción pide unir los tres bloques por su fecha).
        def fecha_reg(reg: str) -> str:
            return str((fichas.get(reg) or {}).get("fecha_publicacion") or tesis[reg][1].get("fecha_publicacion")
                       or "9999")[:10]
        for reg in sorted(regs_xml, key=lambda g: (fecha_reg(g), g)):
            pid, pl = tesis[reg]
            r = fichas.get(reg) or {}
            partes.append(_tesis_xml(pid, pl, r, crono.get(r.get("crono_id") or ""), reg))
        partes.append("</recepcion_mx>")

    if entradas:
        partes.append('<cronologia orden="cronologico">')
        partes += [_entrada_xml(e, docs_crono.get(e["id"])) for e in entradas]
        partes.append("</cronologia>")

    # Un artículo que ya entró como [Doc ID] de una reforma de la cronología
    # (el 19 con la del 31-dic-2024, el 105 con la del 31-oct-2024) no se
    # repite: mismo punto del acervo, mismo texto.
    ya_crono = set(docs_crono.values())
    normas = {pid: c for pid, c in normas.items() if pid not in ya_crono}
    if normas:
        partes.append("<constitucion_mx>")
        partes.append("<!-- La norma mexicana vigente con la que convive la línea. Transcríbela de su "
                      "[Doc ID]; si hay reforma posterior a la sentencia interamericana, dilo. -->")
        for pid, c in normas.items():
            fr = f' fecha_reforma="{_esc(c["fecha_reforma"])}"' if c.get("fecha_reforma") else ""
            partes.append(f'<articulo id="{_esc(pid)}" ref="{_esc(c.get("ref"))}"{fr}>'
                          f"«{_esc(c.get('extracto'))}»" + (f" {_esc(c['nota'])}" if c.get("nota") else "")
                          + f" [Doc ID: {_esc(pid)}]</articulo>")
        partes.append("</constitucion_mx>")

    if sel.get("actual_mx") and (f.get("fase_actual") or {}).get("pendientes"):
        partes.append(f"<pendientes>{_esc(f['fase_actual']['pendientes'])}</pendientes>")
    cortes = _cortes_xml(f)
    if cortes:
        partes.append(cortes)
    # La nota larga de vigente_al (qué sentencias de 2024-2026 no se leyeron)
    # la resume ahora <cortes>; aquí sólo la fecha, y sólo de la Corte IDH
    # (en solo_mx no hay nada suyo: basta <cortes>).
    if not solo_mx:
        partes.append(f"<vigencia>Según lo ingerido hasta {_esc(vig)} en la Corte IDH; lo mexicano se fecha con "
                      "los cortes de cada fuente.</vigencia>")
    partes.append("</linea_jurisprudencial>")
    return "\n".join(partes)


# ═══════════════════════════════════════════════════════════════ seguimiento

_RX_ES_SEGUIMIENTO = re.compile(r"p[aá]rr|¶|§|resolutivo|considerando|voto|\bpara\.", re.I)
_RX_LLAVE_EN_MARCADOR = re.compile(r'"llave"\s*:\s*"([^"|]+\|[^"|]+\|[^"]*)"')
_MARCAS_DOCUMENTO = ("DOCUMENTO ADJUNTO:", "DOCUMENTO_INICIO", "SENTENCIA_INICIO", "AUDITAR_SENTENCIA")


def previo_de_historial(mensajes: Sequence[Any], pregunta: str) -> Optional[str]:
    """La última llave citada en la conversación, para «¿y el párrafo 125?»
    (revisión B.7): la reescritura del hilo está pensada para leyes y tesis y
    puede perder el caso.

    Sólo se busca si la pregunta de ahora es corta y habla de párrafos o votos
    (el resolvedor sólo hereda en ese caso). Primero la pregunta anterior del
    abogado —lo que ÉL citó—; si no nombró caso, la última llave de los
    marcadores de fuentes de la respuesta anterior (`"llave": …`)."""
    if not pregunta or len(pregunta) > 400 or not _RX_ES_SEGUIMIENTO.search(pregunta):
        return None
    try:
        import coidh_catalogo
    except Exception:
        return None
    anteriores = list(mensajes or [])[:-1]
    for m in reversed(anteriores):
        if _g(m, "role") != "user":
            continue
        completo = str(_g(m, "content") or "")
        texto = completo[:TOPE_PREGUNTA]
        # Un escrito adjunto NO da caso previo (revisión B.8, 25-sep-2026):
        # tras subir una demanda que cita «Caso Almonacid…, párr. 124»,
        # «¿qué dice el párrafo 5?» pregunta por el párrafo 5 del ESCRITO, y
        # sin esto se heredaba Almonacid y entraba su ¶5. Los mismos
        # marcadores con que chat_endpoint reconoce documento y sentencia.
        if any(k in completo for k in _MARCAS_DOCUMENTO):
            return None
        try:
            res = [r for r in coidh_catalogo.resolver_citas_coidh(texto) if r.get("doc_id")]
        except Exception:
            res = []
        if res:
            r = res[-1]
            return (r.get("llaves") or [None])[-1] or r["doc_id"]
        break                                  # sólo la pregunta inmediata anterior
    for m in reversed(anteriores):
        if _g(m, "role") == "assistant":
            llaves = _RX_LLAVE_EN_MARCADOR.findall(str(_g(m, "content") or ""))
            return llaves[-1] if llaves else None
    return None


# ═══════════════════════════════════════════════════════════════ /document-full

async def documento_completo(qdrant, highlight_id: Optional[str] = None,
                             doc_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """La resolución entera, por `orden` (plan §3.4: arregla el orden de
    /document-full, que reconstruía por `chunk_index`), con la URL oficial.

    Un texto por punto y en el mismo orden, para que `highlight_chunk_index`
    siga contando puntos; las ventanas de un párrafo partido se recortan en su
    solape (60 tokens) para no repetir texto. None si no es de la Corte IDH o
    si la colección no está."""
    from qdrant_client.models import FieldCondition, Filter, MatchValue
    try:
        if highlight_id and not doc_id:
            pts = await qdrant.retrieve(collection_name=COLECCION, ids=[highlight_id], with_payload=True)
            if not pts:
                return None
            doc_id = (pts[0].payload or {}).get("doc_id")
        if not doc_id:
            return None
        todos, sig = [], None
        while True:
            pts, sig = await qdrant.scroll(
                collection_name=COLECCION,
                scroll_filter=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]),
                limit=256, offset=sig, with_payload=True, with_vectors=False)
            todos.extend(pts or [])
            if sig is None or not pts or len(todos) > 6000:
                break
    except Exception as e:
        print(f"   ⚖️ COIDH /document-full: {type(e).__name__}; se intenta la ruta de siempre")
        return None
    if not todos:
        return None
    todos.sort(key=lambda p: ((p.payload or {}).get("orden") or 0, (p.payload or {}).get("sub") or 0))
    textos: List[str] = []
    hi = None
    previo_seg = None
    previo_llave, previo_txt = None, ""
    for i, p in enumerate(todos):
        pl = p.payload or {}
        cuerpo = (pl.get("texto_raw") or pl.get("texto") or "").strip()
        if pl.get("llave") == previo_llave and previo_txt:
            # Ventana siguiente del mismo párrafo: fuera el solape.
            palabras = cuerpo.split()
            cola = " ".join(previo_txt.split()[-120:])
            for k in range(min(len(palabras), 100), 7, -1):
                if " ".join(palabras[:k]) in cola:
                    cuerpo = " ".join(palabras[k:])
                    break
        seg_id = (pl.get("seg"), pl.get("voto_autor"))
        cab = ""
        if seg_id != previo_seg:
            if pl.get("seg") == "voto":
                cab = f"{(pl.get('tipo_voto') or 'voto').upper()} DEL JUEZ {(_nombre_autor(pl.get('voto_autor')) or '').upper()}\n"
            elif pl.get("seg") == "resolutivos":
                cab = f"PUNTOS RESOLUTIVOS{(' — ' + pl['seccion']) if pl.get('seccion') else ''}\n"
            elif pl.get("seg") == "considerandos":
                cab = "CONSIDERANDO:\n"
        num = pl.get("parrafo")
        pref = f"{num}. " if num is not None and not (pl.get("llave") == previo_llave) else ""
        textos.append(cab + pref + cuerpo)
        previo_seg, previo_llave, previo_txt = seg_id, pl.get("llave"), (pl.get("texto_raw") or "")
        if highlight_id and str(p.id) == str(highlight_id):
            hi = i
    base = todos[hi if hi is not None else 0]
    meta = contrato(base.id, base.payload or {})
    return dict(
        origen=meta["origen"], titulo=meta["origen"], tipo=meta.get("tipo"),
        texto_completo="\n\n".join(textos), total_chunks=len(todos), highlight_chunk_index=hi,
        # El botón «PDF» de este modal es un ENLACE que se abre en otra
        # pestaña, en el navegador del abogado, donde la Corte sí abre: va a
        # corteidh.or.cr, como antes (revisión del 25-sep-2026). La copia de
        # legal-docs es para DIBUJAR en el visor, no para enlazar; viaja en
        # `metadata.pdf_url` con su sha1 por si la app la necesita.
        source_doc_url=meta.get("url_oficial") or meta.get("pdf_url"),
        metadata={**{k: v for k, v in publico(meta).items()
                     if k in ("caso", "serie", "fecha", "url_oficial", "pdf_url", "pdf_sha1", "llave", "pagina",
                              "parrafo", "ancla", "cita_canonica", "seg", "voto_autor", "tipo", "silo")
                     and v is not None},
                  "doc_id": doc_id},
    )
