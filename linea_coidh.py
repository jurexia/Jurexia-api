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
tal como lo escribe `scripts/ingesta_coidh.py`. HOY NO EXISTE: se crea después
con permiso de David. Mientras no exista, todo esto devuelve vacío, lo dice en
el registro y la consulta sigue exactamente igual que antes.
"""
from __future__ import annotations

import hashlib
import html
import json
import os
import re
import unicodedata
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

COLECCION = "coidh"          # el alias; la física es coidh_parrafos_p1 (ingesta_coidh.py)
SILO = "coidh"               # = COLECCION: _parse_results y /cita usan el nombre consultado
RUTA_LINEAS = Path(__file__).resolve().parent / "datos" / "lineas_coidh.json"

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
    # `alias_amplios` («inconvencional») NO se usa: la ficha pide medir cuánto
    # abre la puerta antes de activarlo (es frecuente en tesis mexicanas).
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
        r"|20(2[3-9]|3\d))\b"),
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
# recepción. No abre la puerta por sí solo.
_RX_MEXICO = re.compile(r"\b(mexic\w*|scjn|suprema\s+corte|varios\s+912|ct\s*293|contradiccion\s+de\s+tesis\s+293)\b")


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
    mercantil o civil tampoco (`excluir` del tema)."""
    if not texto or not texto.strip():
        return None
    p = _plegar(texto[:TOPE_PREGUNTA])
    for fid, f in figuras().items():
        casa_figura = any(rx.search(p) for rx in _alias_rx(fid))
        temas = []
        for tid in (f.get("temas_mx") or {}):
            dispara, excluye = _tema_rx(fid, tid)
            if any(rx.search(p) for rx in dispara) and not any(rx.search(p) for rx in excluye):
                temas.append(tid)
        if not casa_figura and not temas:
            continue
        fases: List[str] = []
        if casa_figura:
            ejes = [f"eje:{k}" for k, rx in _RX_EJES.items() if rx.search(p)]
            fases = [k for k, rx in _RX_FASES.items() if rx.search(p)] or (["evolucion"] if ejes else ["concepto"])
            fases += ejes + (["mexico"] if _RX_MEXICO.search(p) else [])
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


def seleccionar(fid: str, fases: Sequence[str]) -> Dict[str, Any]:
    """Qué se trae para (figura, fases): hitos, registros de la SCJN,
    artículos constitucionales y supervisiones. Sin red; se puede probar solo.

    Con presupuesto (medido con cl100k el 25-sep-2026: ~230 tokens por hito):
    el núcleo del origen siempre; los ejes pedidos, enteros; y un cupo por
    fase —«actual» toma lo más reciente, las demás lo que más aporta según la
    arista—, hasta MAX_HITOS. Después, orden cronológico."""
    f = figuras().get(fid) or {}
    por_llave = _hitos_por_llave(fid)
    temas = [x[5:] for x in fases if x.startswith("tema:")]
    ejes = [x[4:] for x in fases if x.startswith("eje:")]
    mexico = "mexico" in fases
    reales = [x for x in fases if x in _FASES_HITO]

    elegidos: Dict[str, int] = {}          # llave → prioridad (menor = antes)
    registros: List[str] = []
    constitucion: List[str] = []
    supervisiones: List[Dict[str, Any]] = []
    advertencias: List[str] = []
    todos = f.get("hitos") or []

    def fase_de(h):
        return (h.get("fase") or "").lower()

    if reales:
        for ll in NUCLEO:
            if ll in por_llave:
                elegidos[ll] = -1
        for eje in ejes:
            rx_eje = _RX_EJES.get(eje)
            for h in todos:
                # Del eje, todos; de otra fase, sólo lo que AMPLÍA el eje según
                # su aporte: Santo Domingo ¶142 está en «complementariedad»
                # pero es el de «todas las autoridades y órganos» (oro 5).
                if fase_de(h) == eje or (h.get("arista") in ("amplía", "origina") and rx_eje is not None
                                         and rx_eje.search(_plegar(h.get("aporte") or ""))):
                    elegidos.setdefault(h["llave"], 0)
        libres = max(0, MAX_HITOS - len(elegidos))
        cupo = max(2, libres // max(1, len(reales)))
        for x in reales:
            quiere = set(_FASES_HITO[x])
            if not mexico:
                quiere -= {"contenido (méxico)"}
            cands = [h for h in todos if fase_de(h) in quiere and h["llave"] not in elegidos
                     # Las reiteraciones pesan poco… salvo en el estado actual,
                     # donde que la Corte siga diciéndolo en 2024-2025 ES la noticia.
                     and (h.get("arista") != "reitera" or x == "actual")]
            if x == "actual":
                cands.sort(key=lambda h: h.get("fecha") or "", reverse=True)
                if not mexico:     # sin México en la pregunta, primero lo interamericano
                    cands.sort(key=lambda h: "(méxico)" in fase_de(h))
            else:
                cands.sort(key=lambda h: (_PESO_ARISTA.get(h.get("arista"), 5), h.get("fecha") or ""))
            for h in cands[:cupo]:
                elegidos.setdefault(h["llave"], 1)
        # Recepción en México: Pleno y Salas (los colegiados van con sus temas).
        for r in f.get("recepcion_mx") or []:
            if r.get("en_acervo") and r.get("instancia") in ("Pleno", "Primera Sala", "Segunda Sala") \
                    and r.get("relacion") in ("recibe", "acota", "precisa"):
                registros.append(str(r["registro"]))
        registros = registros[:(8 if mexico else 6)]
        if mexico:
            constitucion += list(_CONSTITUCION_MEXICO)

    for tid in temas:
        t = (f.get("temas_mx") or {}).get(tid) or {}
        for ll in t.get("hitos") or []:
            if ll in por_llave:
                elegidos[ll] = min(elegidos.get(ll, 0), 0)
        for x in t.get("extra") or []:
            elegidos.setdefault(x["llave"], 1)
        registros = [str(r) for r in t.get("registros") or []] + registros
        constitucion = list(t.get("constitucion") or []) + constitucion
        for s in t.get("supervision") or []:
            if s.get("doc_id") not in {x.get("doc_id") for x in supervisiones}:
                supervisiones.append(s)
        if t.get("advertencia"):
            advertencias.append(t["advertencia"])

    # El tope recorta lo de cupo, nunca el núcleo, los ejes ni los temas; y a
    # cada fase pedida le deja al menos 2 («¿quiénes están obligados HOY?»
    # llena el eje de sujetos y aun así trae lo más reciente).
    fijos = [ll for ll in elegidos if elegidos[ll] <= 0]
    resto = sorted((ll for ll in elegidos if elegidos[ll] > 0),
                   key=lambda ll: (elegidos[ll], por_llave[ll].get("fecha") or "", ll))
    orden = fijos + resto[:max(MAX_HITOS - len(fijos), 2 * len(reales))]
    hitos = sorted((por_llave[ll] for ll in orden), key=lambda h: (h.get("fecha") or "", _orden_llave(h["llave"])))
    return dict(hitos=hitos, registros=list(dict.fromkeys(registros))[:MAX_TESIS],
                constitucion=list(dict.fromkeys(constitucion)), supervisiones=supervisiones,
                advertencias=advertencias, temas=temas, fases=reales, ejes=ejes, mexico=mexico)


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
    (y del resto de SearchResult). `pdf_url` = `url_oficial`, SIN «#page»: la
    página viaja aparte (con #page pegado, la app Android dejaba de reconocer
    el PDF, plan §3.8). `parrafo` va como texto; `voto_autor`, con nombre."""
    url = (pl.get("url_oficial") or pl.get("pdf_url") or "").split("#")[0] or None
    doc_id = str(pl.get("doc_id") or "")
    parrafo = pl.get("parrafo")
    return dict(
        id=str(pid), score=score,
        texto=pl.get("texto") or pl.get("texto_raw") or "",
        ref=pl.get("ref"),
        origen=_origen(doc_id, pl.get("caso"), pl.get("fecha")),
        jurisdiccion="Corte Interamericana de Derechos Humanos",
        entidad=None, silo=SILO, pdf_url=url,
        tipo=pl.get("tipo"), url_oficial=url,
        pagina=int(pl["pagina"]) if isinstance(pl.get("pagina"), (int, float)) else None,
        parrafo=str(parrafo) if parrafo is not None else None,
        seg=pl.get("seg"), voto_autor=_nombre_autor(pl.get("voto_autor")),
        caso=pl.get("caso"), serie=_serie_rotulo(pl.get("serie"), pl.get("serie_num")),
        fecha=pl.get("fecha"), ancla=pl.get("ancla"), cita_canonica=pl.get("cita_canonica"),
        llave=pl.get("llave"), rol_coidh=rol, pdf_sha1=pl.get("pdf_sha1"),
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
    return dict(
        id=ficha_id(doc_id), score=1.0, texto=texto,
        ref=("Ficha del catálogo" if parrafo is None else f"Párr. {parrafo} (no ingerido)"),
        origen=_origen(doc_id, caso, d.get("fecha")),
        jurisdiccion="Corte Interamericana de Derechos Humanos", entidad=None, silo=SILO,
        pdf_url=d.get("url_oficial"), tipo=_tipo_de_doc(doc_id, seg), url_oficial=d.get("url_oficial"),
        pagina=None, parrafo=str(parrafo) if parrafo is not None else None, seg=seg,
        voto_autor=voto_autor, caso=caso, serie=serie, fecha=d.get("fecha"), ancla=None,
        cita_canonica=cita, llave=llave or doc_id, rol_coidh=rol,
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
    url = h.get("url_oficial")
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
        pdf_url=url, tipo=_tipo_de_doc(doc_id, seg), url_oficial=url,
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


async def traer_linea(qdrant, deteccion: Tuple[str, List[str]], coleccion_tesis: str,
                      coleccion_constitucion: str = "bloque_constitucional",
                      tesis_a_dict: Optional[Callable[[str, Dict[str, Any], str], Dict[str, Any]]] = None,
                      norma_a_dict: Optional[Callable[[str, Dict[str, Any], str], Dict[str, Any]]] = None
                      ) -> Optional[Dict[str, Any]]:
    """La línea para (figura, fases): hitos por llave, recepción en México por
    registro y los artículos constitucionales que conviven con ella.

    Devuelve {xml, docs, n, faltan, figura, fases, temas}, o None si no hay
    nada o si la colección no existe (hoy no existe). Nunca lanza: una línea
    que no llega no puede costar la consulta.

    `docs` son dicts con los campos de SearchResult: los hitos (silo «coidh»,
    con el contrato) y, si se pasan los conversores de main.py, las tesis y
    los artículos traídos, para que su [Doc ID] resuelva y el sello los vea."""
    try:
        fid, fases = deteccion
        f = figuras().get(fid)
        if not f:
            return None
        sel = seleccionar(fid, fases)
        if not sel["hitos"] and not sel["registros"]:
            return None
        try:
            puntos = await traer_por_llaves(qdrant, [h["llave"] for h in sel["hitos"]])
        except Exception as e:
            print(f"   ⚖️ COIDH: la colección «{COLECCION}» no responde ({type(e).__name__}): "
                  "la línea no entra y la consulta sigue igual")
            return None

        hitos: List[Dict[str, Any]] = []
        faltan: List[str] = []
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
            hitos.append(d)
        supervisiones = [dict(ficha_supervision(s), ingerido=False, _sup=s) for s in sel["supervisiones"]]

        tesis = await traer_tesis(qdrant, coleccion_tesis, sel["registros"])
        normas: Dict[str, Dict[str, Any]] = {}
        ids_norma = {c["qdrant_id"]: c for c in f.get("constitucion_mx") or []
                     if c.get("ref") in sel["constitucion"] and c.get("qdrant_id") and c.get("en_acervo")}
        if ids_norma:
            normas = await traer_por_ids(qdrant, coleccion_constitucion, list(ids_norma))

        docs = [publico(d) for d in hitos + supervisiones]
        if tesis_a_dict:
            docs += [tesis_a_dict(pid, pl, coleccion_tesis) for pid, pl in tesis.values()]
        if norma_a_dict:
            docs += [norma_a_dict(pid, normas[pid], coleccion_constitucion) for pid in normas]
        xml = bloque_xml(fid, sel, hitos, supervisiones, tesis, {pid: ids_norma[pid] for pid in normas})
        return dict(xml=xml, docs=docs, n=len(docs), faltan=faltan, figura=fid, fases=sel["fases"],
                    temas=sel["temas"], hitos=[h["llave"] for h in sel["hitos"]])
    except Exception as e:
        print(f"   ⚖️ COIDH: la línea falló ({type(e).__name__}: {str(e)[:120]}); la consulta sigue sin ella")
        return None


# ═══════════════════════════════════════════════════════════════ los bloques

_INSTRUCCION_CITA = (
    "CÓMO SE CITA LA CORTE IDH (25-sep-2026):\n"
    "· Sentencia: «Corte IDH. Caso X Vs. Y. [Excepciones…, Fondo…]. Sentencia de [fecha]. "
    "Serie C No. N, párr. P [Doc ID: uuid]». La cita exacta viene en <cita>: cópiala y añade el [Doc ID].\n"
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
    """Un hito, en lo justo (~230 tokens): el caso, la serie y el tipo ya van
    en la <cita>, y el id una sola vez, en su [Doc ID]."""
    h = d.get("_hito") or {}
    attrs = [f'fecha="{_esc(d.get("fecha"))}"', f'fase="{_esc(h.get("fase"))}"', f'arista="{_esc(h.get("arista"))}"']
    for k, nombre in (("parrafo", "parrafo"), ("pagina", "pagina"), ("voto_autor", "voto")):
        if d.get(k) not in (None, ""):
            attrs.append(f'{nombre}="{_esc(d[k])}"')
    attrs.append(f'ingerido="{"sí" if d.get("ingerido") else "no"}"')
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


def bloque_xml(fid: str, sel: Dict[str, Any], hitos: Sequence[Dict[str, Any]],
               supervisiones: Sequence[Dict[str, Any]] = (),
               tesis: Optional[Dict[str, Tuple[str, Dict[str, Any]]]] = None,
               normas: Optional[Dict[str, Dict[str, Any]]] = None) -> str:
    """<linea_jurisprudencial>: los hitos en orden CRONOLÓGICO, cada uno con
    caso, fecha, párrafo, página, aporte, extracto y [Doc ID]; la precisión
    histórica del origen; tensiones; recepción en México (registros de la
    SCJN); y «según lo ingerido hasta [fecha]». Se añade después del contexto
    normal (patrón de la doctrina): la jerarquía rompería la cronología."""
    f = figuras().get(fid) or {}
    tesis = tesis or {}
    normas = normas or {}
    vig = f.get("vigente_al") or lineas().get("generado") or ""
    temas = sel.get("temas") or []
    reales = sel.get("fases") or []
    alcance = ("línea completa" if reales else "hitos del tema, no la línea entera")
    partes = [f'<linea_jurisprudencial figura="{_esc(f.get("nombre") or fid)}" '
              f'fases="{_esc(",".join(reales))}" temas="{_esc(",".join(temas))}" '
              f'alcance="{_esc(alcance)}" curada="sí" vigente_al="{_esc(vig)}">',
              "<!-- INSTRUCCIÓN LÍNEA JURISPRUDENCIAL (Corte IDH): "
              + ("esta es la línea curada y verificada contra el PDF oficial de cada resolución. Cuéntala en "
                 "este orden: expresión (votos de García Ramírez; el control lo ejerce la Corte IDH) → "
                 "formulación en pleno (Almonacid ¶124) → ex officio (Cesados ¶128) → ejes (quién está "
                 "obligado, parámetro, efectos, forma) → estado actual → tensiones → recepción en México. "
                 if reales else
                 "éstos son SÓLO los hitos de la Corte IDH que tocan el tema mexicano de la pregunta, "
                 "verificados contra el PDF oficial; no cuentes la línea entera. Preséntalos después de la "
                 "norma y la jurisprudencia mexicanas, con sus tensiones. ")
              + f"Del estado actual di «según lo ingerido hasta {vig}», nunca «hoy». ingerido=\"no\": el "
              "texto completo no está en Iurexia, sólo su extracto verificado; cita igual su [Doc ID].\n"
              + _INSTRUCCION_CITA + " -->"]
    if reales:
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
            claves.append("mexico")
        for k in dict.fromkeys(claves):
            if resumen.get(k):
                partes.append(f'<resumen parte="{_esc(k)}">{_esc(resumen[k])}</resumen>')
    for adv in sel.get("advertencias") or []:
        partes.append(f"<advertencia>{_esc(adv)}</advertencia>")

    partes.append('<hitos orden="cronologico">')
    partes += [_hito_xml(d) for d in hitos]
    partes.append("</hitos>")

    for s in supervisiones:
        sup = s.get("_sup") or {}
        partes.append(f'<supervision id="{_esc(s["id"])}" fecha="{_esc(sup.get("fecha"))}" '
                      f'caso="{_esc(sup.get("caso"))}" pagina="{_esc(sup.get("pagina"))}" ingerido="no">\n'
                      f"<estado>{_esc(sup.get('estado'))}</estado>\n"
                      + (f"<extracto>«{_esc(sup.get('extracto'))}»</extracto>\n" if sup.get("verificado") else "")
                      + f"<cita>{_esc(s.get('cita_canonica'))} [Doc ID: {_esc(s['id'])}]</cita>\n</supervision>")

    # Tensiones: con la línea, salvo que sólo se pregunte el origen; con un
    # tema, las que tocan sus hitos (la de prisión preventiva, la del arraigo).
    ten = f.get("tensiones") or []
    if reales:
        if reales == ["origen"]:
            ten = []
    else:
        mias = {d.get("llave") for d in hitos}
        ten = [t for t in ten if mias & set((t.get("a_favor") or []) + (t.get("en_contra") or []))]
    if ten:
        partes.append("<tensiones>")
        for t in ten:
            regs = ", ".join(t.get("registros") or [])
            partes.append(f'<tension tema="{_esc(t.get("tema"))}">{_esc(t.get("resumen"))}'
                          + (f" (registros: {_esc(regs)})" if regs else "") + "</tension>")
        partes.append("</tensiones>")

    if tesis:
        fichas = {str(r.get("registro")): r for r in f.get("recepcion_mx") or []}
        partes.append("<recepcion_mx>")
        partes.append("<!-- Criterios de la SCJN y de tribunales mexicanos: son derecho mexicano y se citan "
                      "como cualquier tesis, con su registro y su [Doc ID]. -->")
        for reg in sel.get("registros") or []:
            if reg not in tesis:
                continue
            pid, pl = tesis[reg]
            r = fichas.get(reg) or {}
            clave = pl.get("clave_tesis") or pl.get("numero_tesis") or pl.get("tesis") or r.get("clave") or ""
            rubro = (pl.get("rubro") or r.get("rubro_abreviado") or "").strip()
            partes.append(f'<tesis id="{_esc(pid)}" registro="{_esc(reg)}" clave="{_esc(clave)}" '
                          f'instancia="{_esc(pl.get("instancia") or r.get("instancia"))}" '
                          f'relacion="{_esc(r.get("relacion"))}">{_esc(rubro)}'
                          + (f" — {_esc(r.get('porque'))}" if r.get("porque") else "")
                          + f" [Doc ID: {_esc(pid)}]</tesis>")
        partes.append("</recepcion_mx>")

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

    nota_vig = f.get("vigente_al_nota") or ""
    partes.append(f"<vigencia>Según lo ingerido hasta {_esc(vig)}. {_esc(nota_vig)}</vigencia>")
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
        source_doc_url=meta.get("url_oficial"),
        metadata={**{k: v for k, v in publico(meta).items()
                     if k in ("caso", "serie", "fecha", "url_oficial", "llave", "pagina", "parrafo", "ancla",
                              "cita_canonica", "seg", "voto_autor", "tipo", "silo") and v is not None},
                  "doc_id": doc_id},
    )
