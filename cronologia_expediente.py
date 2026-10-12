"""CRONOLOGÍA DEL EXPEDIENTE.   (12-oct-2026, sólo Platinum)

Inspirada en el «litigation deep dive» de Harvey: de un expediente de
cientos de páginas, la cronología de los hechos con su página, la lista de
cuestiones en disputa y los plazos que corren.

CÓMO SE LEE
-----------
1. Cada PDF se parte en PÁGINAS SIN PERDER SU NÚMERO. `_paginas_de_bytes`
   de main.py tira las páginas vacías y con ellas la numeración: aquí la
   página 37 es la 37 del PDF aunque la 12 esté en blanco, porque es lo que
   el abogado va a abrir. Texto nativo si sirve (`texto_nativo_sirve`), si no
   el OCR de siempre (`_texto_de_bytes`: caché por huella y Azure), que
   rotula cada página con `[[PÁGINA n]]`.
2. Las páginas se mandan a gpt-6-luna por LOTES de un mismo documento, con
   la última página del lote anterior como contexto. Pide JSON: hechos con
   fecha, cuestiones, plazos y los datos del asunto.
3. NADA ENTRA SIN ESTAR ESCRITO EN SU PÁGINA. Cada hecho trae una cita
   literal; la cita tiene que estar en el texto de la página que dice (o, si
   el modelo se equivocó de página, en otra del mismo lote, y entonces se
   corrige la página). La fecha tiene que constar en esa página o en las de
   al lado, en letra o en cifras (`fechas_en_autos.fechas_de`). Lo que no se
   verifica se descarta y se cuenta.
4. Los plazos, como en /agenda/extraer: cita literal en la página, el número
   de la cita igual al del plazo, código que rige (`regla_de`, que es
   `_agenda_regla`) y vencimiento con `plazos_calculo`. La fecha base sólo
   se usa si consta en la página.
5. Las cuestiones de todos los lotes se agrupan en una última llamada; cada
   grupo conserva las páginas verificadas de las que salió.

Este módulo no sabe de usuarios, de cobros ni de Supabase: recibe bytes y
funciones, y devuelve el resultado con su costo medido.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import re
import time
import unicodedata
from difflib import SequenceMatcher
from typing import Awaitable, Callable, Optional

import fechas_en_autos as _fa
import plazos_calculo as _pc

MODELO = os.getenv("CRONOLOGIA_MODELO") or os.getenv("BUSCAR_MODEL", "gpt-6-luna")
PAGINAS_POR_LOTE = max(2, int(os.getenv("CRONOLOGIA_PAGINAS_POR_LOTE", "12")))
CARACTERES_POR_LOTE = max(8000, int(os.getenv("CRONOLOGIA_CARACTERES_POR_LOTE", "40000")))
CONCURRENCIA = max(1, int(os.getenv("CRONOLOGIA_CONCURRENCIA", "6")))
MAX_PAGINAS = int(os.getenv("CRONOLOGIA_MAX_PAGINAS", "1500"))

# Precios para medir lo que cuesta cada expediente (USD).
USD_ENTRADA_M = float(os.getenv("CRONOLOGIA_USD_ENTRADA_M", "0.10"))   # gpt-6-luna, por millón
USD_SALIDA_M = float(os.getenv("CRONOLOGIA_USD_SALIDA_M", "0.50"))
USD_OCR_PAGINA = 0.0015                                                  # Azure prebuilt-read

_CITA_MIN_PALABRAS = 6


# ── TEXTO ────────────────────────────────────────────────────────────────────
def _compacto(t: str) -> str:
    """Sin acentos, sin mayúsculas, sin nada que no sea letra o cifra. Así la
    misma frase coincide aunque el OCR partiera una palabra con guion al final
    de la línea o metiera dos espacios."""
    t = unicodedata.normalize("NFKD", t or "")
    t = "".join(c for c in t if not unicodedata.combining(c)).lower()
    return re.sub(r"[^a-z0-9]+", "", t)


def _palabras(t: str) -> list:
    t = unicodedata.normalize("NFKD", t or "")
    t = "".join(c for c in t if not unicodedata.combining(c)).lower()
    return re.findall(r"[a-z0-9]+", t)


def _ngramas(pal: list, n: int = 4) -> set:
    return {" ".join(pal[i:i + n]) for i in range(max(0, len(pal) - n + 1))}


class Pagina:
    __slots__ = ("doc", "num", "texto", "_c", "_ng", "_fechas")

    def __init__(self, doc: int, num: int, texto: str):
        self.doc, self.num, self.texto = doc, num, texto or ""
        self._c = self._ng = self._fechas = None

    @property
    def compacto(self) -> str:
        if self._c is None:
            self._c = _compacto(self.texto)
        return self._c

    @property
    def ngramas(self) -> set:
        if self._ng is None:
            self._ng = _ngramas(_palabras(self.texto))
        return self._ng

    @property
    def fechas(self) -> set:
        if self._fechas is None:
            try:
                self._fechas = set(_fa.fechas_de(self.texto).keys())
            except Exception:
                self._fechas = set()
        return self._fechas


def cita_en_pagina(cita: str, pag: Pagina) -> Optional[str]:
    """'exacta' si la cita está tal cual (salvo espacios, acentos y signos);
    'aproximada' si el 85% de sus tramos de cuatro palabras están en la
    página —el modelo enmendó una letra del OCR—; None si no está."""
    pal = _palabras(cita)
    if len(pal) < _CITA_MIN_PALABRAS or not pag.texto:
        return None
    c = _compacto(cita)
    if len(c) >= 25 and c in pag.compacto:
        return "exacta"
    ng = _ngramas(pal)
    if ng and len(ng & pag.ngramas) / len(ng) >= 0.85:
        return "aproximada"
    return None


def fecha_iso(valor) -> Optional[tuple]:
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", str(valor or "").strip())
    if not m:
        return None
    a, mes, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    try:
        dt.date(a, mes, d)
    except ValueError:
        return None
    return (a, mes, d)


# ── LECTURA POR PÁGINAS ──────────────────────────────────────────────────────
def contar_paginas(contenido: bytes) -> int:
    import fitz
    d = fitz.open(stream=contenido, filetype="pdf")
    try:
        return len(d)
    finally:
        d.close()


def _paginas_nativas(contenido: bytes) -> list:
    import fitz
    d = fitz.open(stream=contenido, filetype="pdf")
    try:
        return [p.get_text() for p in d]
    finally:
        d.close()


def partir_por_rotulos(texto: str, total: int) -> Optional[list]:
    """El texto del OCR (rotulado `[[PÁGINA n]]`) puesto en su página. Si el
    documento se leyó por tramos de Azure, cada tramo vuelve a empezar en 1:
    cuando el rótulo retrocede, se suma lo ya visto. None si no hay rótulos
    (el respaldo de Gemini no los pone)."""
    trozos = re.split(r"\[\[PÁGINA (\d+)\]\]\n?", texto or "")
    if len(trozos) < 3:
        return None
    paginas = [""] * max(total, 1)
    base, ultimo = 0, 0
    for i in range(1, len(trozos) - 1, 2):
        n = int(trozos[i])
        if n <= ultimo - base:
            base = ultimo
        num = base + n
        ultimo = max(ultimo, num)
        if num > len(paginas):
            paginas.extend([""] * (num - len(paginas)))
        paginas[num - 1] = (trozos[i + 1] or "").strip()
    return paginas


async def leer_paginas(contenido: bytes, nombre: str, *,
                       texto_nativo_sirve: Callable[[str, int], bool],
                       texto_de_bytes: Callable[[bytes, str], Awaitable[str]],
                       cache_tiene: Callable[[bytes], Awaitable[bool]]) -> dict:
    """{nombre, paginas: [texto por página], total, ocr: nativo|cache|azure|sin_paginas}."""
    nativas = await asyncio.to_thread(_paginas_nativas, contenido)
    total = len(nativas)
    if texto_nativo_sirve("\n\n".join(nativas).strip(), total):
        return {"nombre": nombre, "paginas": nativas, "total": total, "ocr": "nativo"}
    # ¿Se va a pagar Azure? Sólo para medir: el OCR decide solo.
    try:
        en_cache = await cache_tiene(contenido)
    except Exception:
        en_cache = False
    texto = await texto_de_bytes(contenido, nombre)
    paginas = partir_por_rotulos(texto, total)
    if paginas is None:
        # Sin rótulos no hay manera honesta de decir en qué página está cada
        # cosa: el documento entero queda como su primera página y la pantalla
        # lo advierte.
        return {"nombre": nombre, "paginas": [texto or ""] + [""] * max(total - 1, 0),
                "total": total, "ocr": "sin_paginas"}
    return {"nombre": nombre, "paginas": paginas, "total": max(total, len(paginas)),
            "ocr": "cache" if en_cache else "azure"}


# ── LOTES ────────────────────────────────────────────────────────────────────
def armar_lotes(docs: list, paginas_por_lote: int = PAGINAS_POR_LOTE,
                caracteres: int = CARACTERES_POR_LOTE) -> list:
    """[(doc, [números de página], número de la página de contexto o None)].
    Nunca mezcla documentos; las páginas vacías no viajan."""
    lotes = []
    for d, doc in enumerate(docs):
        actual, chars, previa = [], 0, None
        for i, texto in enumerate(doc["paginas"], 1):
            t = (texto or "").strip()
            if len(t) < 15:
                continue
            largo = min(len(t), 9000)
            if actual and (len(actual) >= paginas_por_lote or chars + largo > caracteres):
                lotes.append((d, actual, previa))
                previa = actual[-1]
                actual, chars = [], 0
            actual.append(i)
            chars += largo
        if actual:
            lotes.append((d, actual, previa))
    return lotes


INSTRUCCION = """Eres analista de expedientes judiciales mexicanos. Recibes páginas de un expediente (texto de OCR que puede traer errores). Cada página empieza con su marcador ⟦PÁGINA n⟧, donde n es el número de página del PDF. La página marcada como CONTEXTO sólo sirve para entender la continuidad: no extraigas nada de ella.

Responde SOLO con un objeto JSON con esta forma:
{"hechos":[{"fecha":"AAAA-MM-DD","hecho":"qué ocurrió, en una frase clara","actuacion":"tipo de actuación o documento: demanda, auto admisorio, emplazamiento, contestación, reconvención, audiencia, desahogo de pruebas, alegatos, sentencia, recurso, notificación, promoción, oficio, dictamen, convenio, contrato, etc.","actor":"quién la realiza, promueve, dicta o a quién se refiere","pagina":n,"foja":"número de foja si está estampado en la página; si no, cadena vacía","relevancia":"alta|media|baja","cita":"copia EXACTA y continua, de 8 a 35 palabras, del pasaje de ESA página que prueba el hecho y su fecha"}],
"cuestiones":[{"cuestion":"punto en disputa, agravio o concepto de violación, en una frase","sostiene":"quién lo plantea","postura":"qué sostiene","contraparte":"qué responde la otra parte o qué resolvió la autoridad, si consta en estas páginas; si no, cadena vacía","pagina":n,"cita":"copia EXACTA de 8 a 35 palabras de esa página"}],
"plazos":[{"acto":"lo que debe hacer la parte","parte":"a quién se le concede","cantidad":9,"unidad":"dias|horas|meses|anos","tipo_dias":"habiles|naturales|no_especifica","fundamento_citado":"artículo y ley si se citan; si no, cadena vacía","texto_literal":"copia EXACTA del pasaje donde está el plazo","pagina":n,"fecha_acuerdo":"AAAA-MM-DD o null","fecha_notificacion":"AAAA-MM-DD o null","tipo_notificacion":"personal|lista|electronica|desconocida","apercibimiento":"la consecuencia si no se hace, si se dice"}],
"datos":{"expediente":"número de expediente si aparece","organo":"juzgado o tribunal","entidad":"estado de la República o Federal","materia":"civil|familiar|penal|laboral|mercantil|administrativo|fiscal|amparo|agrario|otra","partes":["nombres y carácter procesal"]}}

Reglas:
- Hechos: uno por cada actuación (presentación de escritos, acuerdos, notificaciones, audiencias, resoluciones) y por cada hecho con fecha que narren las partes. Si el hecho lo afirma una parte y no es una actuación del órgano, dilo en "hecho" («según la actora, …»). "relevancia" alta = actuaciones que cambian el curso del juicio o hechos centrales de la litis; baja = trámite.
- Toda fecha debe constar completa (día, mes y año) en la página citada, en letra o en cifras. Un hecho sin fecha completa en el texto NO se incluye. No deduzcas fechas.
- "cita" y "texto_literal" se copian carácter por carácter del texto de la página, con sus errores de OCR: no los corrijas, no los resumas, no unas trozos separados.
- Plazos: sólo los que un acuerdo, auto o resolución concede u ordena (no los que una parte menciona en abstracto). "fecha_notificacion" sólo si la página muestra la razón, constancia o cédula de notificación de ESE acuerdo.
- No inventes nada. Si las páginas no tienen hechos, cuestiones o plazos, devuelve listas vacías."""


INSTRUCCION_CUESTIONES = """Recibes cuestiones en disputa extraídas por partes de un mismo expediente judicial mexicano, cada una con su id. Agrupa las que tratan del MISMO punto (aunque estén redactadas distinto) y redacta cada grupo en una sola cuestión. No inventes cuestiones nuevas ni agregues hechos que no estén en las recibidas.
Responde SOLO con JSON: {"cuestiones":[{"cuestion":"el punto en disputa, en una frase","sostiene":"quién lo plantea","postura":"qué sostiene","contraparte":"qué responde la otra parte o qué resolvió la autoridad, si consta","ids":["c1","c7"]}]}
Ordénalas de la más a la menos importante para resolver el asunto. Cada id debe aparecer en un solo grupo."""


def _texto_de_lote(docs: list, d: int, paginas: list, previa: Optional[int]) -> str:
    partes = []
    if previa:
        t = (docs[d]["paginas"][previa - 1] or "").strip()[-2500:]
        partes.append(f"⟦PÁGINA {previa} · CONTEXTO⟧\n{t}")
    for n in paginas:
        t = (docs[d]["paginas"][n - 1] or "").strip()[:9000]
        partes.append(f"⟦PÁGINA {n}⟧\n{t}")
    return f"Documento: {docs[d]['nombre']}\n\n" + "\n\n".join(partes)


# ── MODELO ───────────────────────────────────────────────────────────────────
class Contador:
    def __init__(self):
        self.entrada = self.salida = self.llamadas = self.fallidas = 0

    def sumar(self, uso: dict):
        self.llamadas += 1
        self.entrada += int((uso or {}).get("prompt_tokens") or 0)
        self.salida += int((uso or {}).get("completion_tokens") or 0)

    @property
    def usd(self) -> float:
        return self.entrada * USD_ENTRADA_M / 1e6 + self.salida * USD_SALIDA_M / 1e6


async def llamar_json(cli, sistema: str, usuario: str, contador: Contador,
                      max_tokens: int = 16000, intentos: int = 3) -> dict:
    """Chat Completions de OpenAI en modo JSON. Reintenta los tropiezos (429,
    5xx, cortes, JSON roto); lo que siga fallando sube como excepción."""
    ultimo = None
    for intento in range(1, intentos + 1):
        try:
            r = await cli.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', '')}"},
                json={"model": MODELO, "reasoning_effort": "low",
                      "max_completion_tokens": max_tokens,
                      "response_format": {"type": "json_object"},
                      "messages": [{"role": "system", "content": sistema},
                                   {"role": "user", "content": usuario}]})
            if r.status_code in (429, 500, 502, 503, 504):
                raise RuntimeError(f"OpenAI {r.status_code}")
            r.raise_for_status()
            j = r.json()
            contador.sumar(j.get("usage") or {})
            contenido = j["choices"][0]["message"].get("content") or "{}"
            return json.loads(contenido)
        except Exception as e:
            ultimo = e
            if intento < intentos:
                await asyncio.sleep(1.5 * intento)
    contador.fallidas += 1
    raise RuntimeError(f"el modelo no respondió: {type(ultimo).__name__}: {str(ultimo)[:160]}")


# ── VERIFICACIÓN ─────────────────────────────────────────────────────────────
def _ubicar(cita: str, pag_dicha: Optional[int], candidatas: list, indice: dict, d: int):
    """(página donde de verdad está la cita, cómo se verificó) o (None, None).
    Primero la página que dice el modelo; si no, las demás del lote."""
    orden = ([pag_dicha] if pag_dicha in candidatas else []) + [p for p in candidatas if p != pag_dicha]
    for p in orden:
        pg = indice.get((d, p))
        if pg is None:
            continue
        como = cita_en_pagina(cita, pg)
        if como:
            return p, como
    return None, None


def _fecha_consta(fecha: tuple, d: int, p: int, indice: dict) -> bool:
    for q in (p, p - 1, p + 1):
        pg = indice.get((d, q))
        if pg is not None and fecha in pg.fechas:
            return True
    return False


def _entero(v) -> Optional[int]:
    try:
        return int(str(v).strip())
    except Exception:
        return None


def verificar_lote(ext: dict, d: int, paginas: list, indice: dict, descartes: dict) -> dict:
    """Lo que el modelo sacó de un lote, sólo con lo que está escrito."""
    def descartar(tipo, motivo):
        descartes[tipo] = descartes.get(tipo, 0) + 1
        clave = f"{tipo}:{motivo}"
        descartes["motivos"][clave] = descartes["motivos"].get(clave, 0) + 1

    hechos, cuestiones, plazos = [], [], []
    for h in (ext.get("hechos") or [])[:200]:
        if not isinstance(h, dict):
            continue
        f = fecha_iso(h.get("fecha"))
        if not f:
            descartar("hechos", "sin_fecha")
            continue
        p_dicha = _entero(h.get("pagina"))
        p, como = _ubicar(str(h.get("cita") or ""), p_dicha, paginas, indice, d)
        if p is None:
            descartar("hechos", "cita_no_esta_en_la_pagina")
            continue
        if not _fecha_consta(f, d, p, indice):
            descartar("hechos", "fecha_no_consta")
            continue
        hechos.append({
            "fecha": "%04d-%02d-%02d" % f, "hecho": str(h.get("hecho") or "").strip()[:600],
            "actuacion": str(h.get("actuacion") or "").strip()[:120],
            "actor": str(h.get("actor") or "").strip()[:200],
            "documento": d, "pagina": p, "foja": str(h.get("foja") or "").strip()[:20],
            "relevancia": h.get("relevancia") if h.get("relevancia") in ("alta", "media", "baja") else "media",
            "cita": str(h.get("cita") or "").strip()[:500], "verificacion": como,
            "pagina_corregida": p != p_dicha,
        })
    for c in (ext.get("cuestiones") or [])[:60]:
        if not isinstance(c, dict) or not str(c.get("cuestion") or "").strip():
            continue
        p_dicha = _entero(c.get("pagina"))
        p, como = _ubicar(str(c.get("cita") or ""), p_dicha, paginas, indice, d)
        if p is None:
            descartar("cuestiones", "cita_no_esta_en_la_pagina")
            continue
        cuestiones.append({
            "cuestion": str(c.get("cuestion") or "").strip()[:500],
            "sostiene": str(c.get("sostiene") or "").strip()[:200],
            "postura": str(c.get("postura") or "").strip()[:800],
            "contraparte": str(c.get("contraparte") or "").strip()[:800],
            "documento": d, "pagina": p, "cita": str(c.get("cita") or "").strip()[:500],
            "verificacion": como})
    for t in (ext.get("plazos") or [])[:30]:
        if not isinstance(t, dict):
            continue
        literal = str(t.get("texto_literal") or "")
        p_dicha = _entero(t.get("pagina"))
        p, como = _ubicar(literal, p_dicha, paginas, indice, d)
        cantidad = _entero(t.get("cantidad"))
        if p is None:
            descartar("plazos", "cita_no_esta_en_la_pagina")
            continue
        if cantidad is None or _pc.numero_de_texto(literal) != cantidad:
            descartar("plazos", "numero_no_coincide")
            continue
        plazos.append({**{k: t.get(k) for k in ("acto", "parte", "unidad", "tipo_dias", "fundamento_citado",
                                                 "fecha_acuerdo", "fecha_notificacion", "tipo_notificacion",
                                                 "apercibimiento")},
                       "cantidad": cantidad, "texto_literal": literal.strip()[:600],
                       "documento": d, "pagina": p, "verificacion": como})
    return {"hechos": hechos, "cuestiones": cuestiones, "plazos": plazos,
            "datos": ext.get("datos") if isinstance(ext.get("datos"), dict) else {}}


# ── CONSOLIDACIÓN ────────────────────────────────────────────────────────────
def _parecidos(a: str, b: str) -> float:
    return SequenceMatcher(None, _compacto(a)[:300], _compacto(b)[:300]).ratio()


def deduplicar_hechos(hechos: list) -> list:
    """La página de contexto y los lotes que se tocan sacan el mismo hecho
    dos veces. Mismo día y (misma cita o hecho casi igual) = el mismo."""
    rango = {"alta": 0, "media": 1, "baja": 2}
    fuera: list = []
    for h in sorted(hechos, key=lambda x: (x["fecha"], x["documento"], x["pagina"], rango[x["relevancia"]])):
        repetido = False
        for g in fuera[-40:]:
            if g["fecha"] != h["fecha"]:
                continue
            if _compacto(g["cita"]) == _compacto(h["cita"]) or _parecidos(g["hecho"], h["hecho"]) >= 0.82:
                repetido = True
                if rango[h["relevancia"]] < rango[g["relevancia"]]:
                    g["relevancia"] = h["relevancia"]
                break
        if not repetido:
            fuera.append(h)
    for i, h in enumerate(fuera, 1):
        h["id"] = f"h{i}"
    return fuera


def _consenso(datos: list) -> dict:
    """El dato del asunto que más lotes repiten."""
    from collections import Counter
    fuera = {}
    for clave in ("expediente", "organo", "entidad", "materia"):
        c = Counter(str(x.get(clave) or "").strip() for x in datos if str(x.get(clave) or "").strip())
        fuera[clave] = c.most_common(1)[0][0] if c else ""
    partes = []
    for x in datos:
        for p in (x.get("partes") or [])[:10]:
            p = str(p).strip()
            if p and all(_parecidos(p, q) < 0.8 for q in partes):
                partes.append(p)
    fuera["partes"] = partes[:12]
    return fuera


async def agrupar_cuestiones(cli, cuestiones: list, contador: Contador) -> list:
    """Una lista de cuestiones sin repetidas. Cada grupo lleva las citas
    verificadas de las que salió; si la llamada falla, van sin agrupar."""
    if not cuestiones:
        return []
    for i, c in enumerate(cuestiones, 1):
        c["id"] = f"c{i}"
    por_id = {c["id"]: c for c in cuestiones}

    def _cita(c):
        return {"documento": c["documento"], "pagina": c["pagina"], "cita": c["cita"]}

    try:
        entrada = [{k: c[k] for k in ("id", "cuestion", "sostiene", "postura", "contraparte")}
                   for c in cuestiones[:300]]
        r = await llamar_json(cli, INSTRUCCION_CUESTIONES, json.dumps(entrada, ensure_ascii=False),
                              contador, max_tokens=12000)
        grupos, usados = [], set()
        for g in (r.get("cuestiones") or []):
            ids = [i for i in (g.get("ids") or []) if i in por_id and i not in usados]
            if not ids or not str(g.get("cuestion") or "").strip():
                continue
            usados.update(ids)
            citas = sorted((_cita(por_id[i]) for i in ids), key=lambda x: (x["documento"], x["pagina"]))
            grupos.append({"cuestion": str(g.get("cuestion")).strip()[:500],
                           "sostiene": str(g.get("sostiene") or "").strip()[:200],
                           "postura": str(g.get("postura") or "").strip()[:900],
                           "contraparte": str(g.get("contraparte") or "").strip()[:900],
                           "citas": citas})
        # Lo que el modelo dejó fuera no se pierde.
        for c in cuestiones:
            if c["id"] not in usados:
                grupos.append({k: c[k] for k in ("cuestion", "sostiene", "postura", "contraparte")}
                              | {"citas": [_cita(c)]})
    except Exception as e:
        print(f"   ⚠️ CRONOLOGÍA: no se pudieron agrupar las cuestiones: {e}")
        grupos = [{k: c[k] for k in ("cuestion", "sostiene", "postura", "contraparte")} | {"citas": [_cita(c)]}
                  for c in cuestiones]
    for i, g in enumerate(grupos, 1):
        g["id"] = f"q{i}"
    return grupos


def computar_plazos(plazos: list, datos: dict, indice: dict, docs: list,
                    regla_de: Callable[[str, str, str, str], tuple], hoy: dt.date) -> list:
    fuera, vistos = [], set()
    for t in plazos:
        llave = (_compacto(t["texto_literal"])[:120], t.get("fecha_acuerdo"), t.get("fecha_notificacion"))
        if llave in vistos:
            continue
        vistos.add(llave)
        d, p = t["documento"], t["pagina"]
        avisos = []
        # LA FECHA BASE SÓLO SI CONSTA. Un vencimiento calculado sobre una
        # fecha que la página no dice es un vencimiento inventado.
        base, de_que = None, None
        for campo in ("fecha_notificacion", "fecha_acuerdo"):
            f = fecha_iso(t.get(campo))
            if f and _fecha_consta(f, d, p, indice):
                base, de_que = "%04d-%02d-%02d" % f, campo
                break
        if base is None:
            avisos.append("La fecha de notificación no consta junto al acuerdo: captúrela para calcular el vencimiento.")
        elif de_que == "fecha_acuerdo":
            avisos.append("Se calculó desde la fecha del acuerdo porque no consta la notificación: corríjala con la de su cédula o lista.")
        texto_pag = (docs[d]["paginas"][p - 1] or "")
        try:
            regla, motivo = regla_de(t.get("fundamento_citado") or "", datos.get("materia") or "",
                                     datos.get("entidad") or "", texto_pag)
        except Exception:
            regla, motivo = "LOCAL", "Regla genérica; revise el cómputo con su código."
        tipo_not = t.get("tipo_notificacion") if t.get("tipo_notificacion") in ("personal", "lista", "electronica") else "personal"
        calc = None
        if base and t.get("unidad") in ("dias", "horas", "meses", "anos") and 0 < t["cantidad"] <= 400:
            try:
                calc = _pc.calcular(base, t["cantidad"], t["unidad"], regla, tipo_not,
                                    t.get("tipo_dias") if t.get("tipo_dias") in ("habiles", "naturales") else None)
            except Exception as e:
                avisos.append(f"No se pudo calcular el vencimiento ({type(e).__name__}).")
        venc = (calc or {}).get("vencimiento")
        estado = "sin_computo" if not venc else ("corre" if dt.date.fromisoformat(venc) >= hoy else "vencido")
        fuera.append({**t, "id": f"t{len(fuera) + 1}", "fecha_base": base, "tipo_notificacion": tipo_not,
                      "regla": regla, "regla_motivo": motivo, "calculo": calc, "estado": estado,
                      "avisos": avisos})
    orden = {"corre": 0, "sin_computo": 1, "vencido": 2}
    fuera.sort(key=lambda x: (orden[x["estado"]], (x.get("calculo") or {}).get("vencimiento") or "9999"))
    return fuera


# ── TODO JUNTO ───────────────────────────────────────────────────────────────
async def procesar(docs: list, *, regla_de: Callable, al_progreso: Optional[Callable[[dict], Awaitable]] = None,
                   hoy: Optional[dt.date] = None, ocr_pagadas: int = 0) -> dict:
    """`docs` = [{nombre, paginas: [texto], total, ocr}] ya leídos."""
    import httpx

    t0 = time.time()
    hoy = hoy or dt.date.today()
    indice = {(d, i): Pagina(d, i, t) for d, doc in enumerate(docs) for i, t in enumerate(doc["paginas"], 1)}
    lotes = armar_lotes(docs)
    contador = Contador()
    descartes = {"hechos": 0, "cuestiones": 0, "plazos": 0, "motivos": {}}
    resultados, fallidos = [], []
    hechos_lotes = 0
    sem = asyncio.Semaphore(CONCURRENCIA)

    async def avisar(extra: dict):
        if al_progreso:
            try:
                await al_progreso(extra)
            except Exception as e:
                print(f"   ⚠️ CRONOLOGÍA: progreso sin guardar: {type(e).__name__}")

    await avisar({"fase": "analizando", "lotes_hechos": 0, "lotes_total": len(lotes)})
    async with httpx.AsyncClient(timeout=httpx.Timeout(240, connect=20)) as cli:
        async def uno(lote):
            nonlocal hechos_lotes
            d, paginas, previa = lote
            async with sem:
                try:
                    ext = await llamar_json(cli, INSTRUCCION, _texto_de_lote(docs, d, paginas, previa), contador)
                    resultados.append(verificar_lote(ext, d, paginas, indice, descartes))
                except Exception as e:
                    print(f"   ⚠️ CRONOLOGÍA: lote {docs[d]['nombre']} págs. {paginas[0]}-{paginas[-1]} falló: {e}")
                    fallidos.append({"documento": d, "desde": paginas[0], "hasta": paginas[-1]})
            hechos_lotes += 1
            await avisar({"fase": "analizando", "lotes_hechos": hechos_lotes, "lotes_total": len(lotes)})

        await asyncio.gather(*(uno(l) for l in lotes))
        if lotes and len(fallidos) == len(lotes):
            raise RuntimeError("Ningún lote del expediente se pudo analizar.")

        await avisar({"fase": "consolidando", "lotes_hechos": len(lotes), "lotes_total": len(lotes)})
        hechos = deduplicar_hechos([h for r in resultados for h in r["hechos"]])
        datos = _consenso([r["datos"] for r in resultados])
        cuestiones = await agrupar_cuestiones(cli, [c for r in resultados for c in r["cuestiones"]], contador)
        plazos = computar_plazos([t for r in resultados for t in r["plazos"]], datos, indice, docs, regla_de, hoy)

    paginas_total = sum(int(doc.get("total") or len(doc["paginas"])) for doc in docs)
    usd_ocr = ocr_pagadas * USD_OCR_PAGINA
    return {
        "datos": datos,
        "documentos": [{"nombre": doc["nombre"], "paginas": int(doc.get("total") or len(doc["paginas"])),
                        "ocr": doc.get("ocr"),
                        "sin_texto": sum(1 for t in doc["paginas"] if len((t or "").strip()) < 15)}
                       for doc in docs],
        "hechos": hechos,
        "cuestiones": cuestiones,
        "plazos": plazos,
        "descartes": descartes,
        "lotes_fallidos": fallidos,
        "costo": {"paginas": paginas_total, "lotes": len(lotes), "llamadas": contador.llamadas,
                  "tokens_entrada": contador.entrada, "tokens_salida": contador.salida,
                  "usd_modelo": round(contador.usd, 5), "paginas_ocr_pagadas": ocr_pagadas,
                  "usd_ocr": round(usd_ocr, 5), "usd_total": round(contador.usd + usd_ocr, 5),
                  "segundos": round(time.time() - t0, 1), "modelo": MODELO},
        "generado": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
