#!/usr/bin/env python3
"""AÑADE LEYES FEDERALES QUE NO ESTÁN A `leyes_federales`, SIN TOCAR LAS 72 QUE SÍ.   (24-sep-2026)

POR QUÉ EXISTE
--------------
Un magistrado Pro que trabaja para el Corredor Interoceánico preguntaba por la
Ley Federal de las Entidades Paraestatales y recibía «no me consta el texto».
No era el modelo: la colección tenía 72 leyes de las 316 que publica la Cámara
de Diputados. Faltaban, entre otras, Adquisiciones, Obras Públicas, APP,
Presupuesto, Planeación, Transparencia, Puertos, Ferroviario, Aeropuertos,
Inversión Extranjera, Mercado de Valores y la de Infraestructura Estratégica
del 9-abr-2026.

EL FORMATO SE COPIA, NO SE INVENTA
----------------------------------
El pipeline que llenó la colección (`ingestion_version: v2_terminator`) ya no
existe en ningún repositorio. Se reconstruyó midiendo los 17,739 puntos:

  · payload: texto, texto_raw, ref, ley, titulo, capitulo, seccion, subtitulo,
    jerarquia, materia, entidad, chunk_index, es_transitorio, source_type,
    ingestion_version, cuerpo_legal_oficial, url_pdf, pdf, articulo_num.
  · `texto` = «[ley | titulo | capitulo | seccion | subtitulo]\\n» + texto_raw
    (sólo las partes no vacías) — cierto en el 100 % de los puntos.
  · `jerarquia` = «ley > titulo > capitulo > seccion > ref» — 100 %.
  · vector «dense»: text-embedding-3-small (1536, coseno).
  · vector «sparse»: NO es el BM25 de fastembed. Es `1 + ln(1 + n/T)` por
    palabra de `texto.lower().split()` (con su puntuación pegada), sin las de
    una letra ni 41 palabras vacías, índice `int(md5(palabra)[:8], 16) % 30000`,
    cada término redondeado a 4 decimales y sumado si dos palabras caen en el
    mismo índice. Reproduce 17,737 de 17,739 vectores exactos (los otros dos
    difieren en el cuarto decimal). Hoy el chat busca sólo con el denso
    (`_SOLO_DENSO` en main.py), pero el día que se encienda el híbrido estos
    puntos deben hablar el mismo idioma que sus vecinos.

Lo que NO se copia, a propósito: los defectos. El v2 guardaba «TITULO C» por
«TÍTULO CUARTO», perdía el sufijo de los bis («Artículo 69» para el 69-H) y le
faltan artículos enteros (el 69-B del CFF). Aquí `ref` lleva la designación
completa («Artículo 69-B.») y `articulo_num` el entero, como siempre.

LO QUE NUNCA HACE
-----------------
No borra ni reescribe un solo punto existente: sólo `upsert` de ids nuevos
(md5 de «federales-2026-09|ley|ref|trozo|posición»), y aborta si la ley ya
está en la colección con ese nombre. Cada punto nuevo lleva
`ingesta: "federales-2026-09"` para poder deshacer con `--revertir`.

    python scripts/ingesta_leyes_federales.py --seco                    # cuenta, sin API
    python scripts/ingesta_leyes_federales.py --seco --abrev LFEP LAPP  # sólo esas
    python scripts/ingesta_leyes_federales.py --abrev LFEP LAPP         # embebe y sube
    python scripts/ingesta_leyes_federales.py --revertir --abrev LFEP   # borra lo subido
"""
import argparse
import hashlib
import json
import math
import re
import sys
import time
import uuid
from collections import Counter
from pathlib import Path

COLECCION = "leyes_federales"
MARCA = "federales-2026-09"
MODELO_EMBED = "text-embedding-3-small"
MAX_CHARS = 6000          # el trozo más largo del v2 mide exactamente 6,000
SOLAPE = 400
MIN_ARTICULO = 12         # «Artículo 25.- Derogado» también se cita
LOTE_EMBED, LOTE_UPSERT, PAUSA = 96, 20, 0.35

# ══════════════════════════════════════════════════════════════════════
# VECTOR DISPERSO — el del v2, medido punto por punto
# ══════════════════════════════════════════════════════════════════════
PALABRAS_VACIAS = frozenset(
    "de la el en que los del las se por para al su con no lo un como cuando una este así sin "
    "sobre ser le entre hasta es más les son todo desde pero ya ha hay muy pueden está".split())


def disperso(texto: str) -> tuple[list[int], list[float]]:
    palabras = [t for t in texto.lower().split() if len(t) >= 2 and t not in PALABRAS_VACIAS]
    if not palabras:
        return [], []
    total = len(palabras)
    acumulado: dict[int, float] = {}
    for palabra, n in Counter(palabras).items():
        i = int(hashlib.md5(palabra.encode()).hexdigest()[:8], 16) % 30000
        acumulado[i] = acumulado.get(i, 0.0) + round(1 + math.log(1 + n / total), 4)
    indices = sorted(acumulado)
    return indices, [acumulado[i] for i in indices]


# ══════════════════════════════════════════════════════════════════════
# DEL PDF DE LA CÁMARA A PÁRRAFOS
# ══════════════════════════════════════════════════════════════════════
# Las notas de reforma de la Cámara («Párrafo reformado DOF 01-03-2019») no son
# texto de la ley y el v2 no las guardaba: sólo 118 de 17,739 trozos traen una.
_VERBO_NOTA = r"(?:reformad|adicionad|derogad|recorrid|modificad|publicad|abrogad|reubicad|actualizad)[oa]s?"
_FECHAS_NOTA = (r"(?:[\s,.;y]+(?:(?:DOF\s+)?\d{2}-\d{2}-\d{4}|" + _VERBO_NOTA + r"(?:\s+DOF)?"
                r"|(?:con\s+)?fe\s+de\s+erratas(?:\s+DOF)?|\([^)\n]{0,60}\)|Publicad[oa]\s+[íi]ntegr[oa][^\n]{0,30}?"
                r"|(?:P[áa]rrafo|Art[íi]culo|Fracci[óo]n|Inciso|Cantidades)[^\n]{0,30}?(?=\s" + _VERBO_NOTA + r")))*[\s.,;]*")
# Una nota ocupa el renglón entero: sujeto corto («Párrafo con fracciones
# reformado», «Concepto reformado», o nada: «Derogado DOF 18-11-2015»), verbo,
# DOF y fechas. O es la continuación partida de otra: «21-12-2005. Derogado DOF
# 13-11-2008». Un renglón con texto de la ley después de la fecha no casa.
RE_NOTA_REFORMA = re.compile(
    r"^(?:(?:[A-ZÁÉÍÓÚ][^\n]{0,60}?\s)?" + _VERBO_NOTA + r"[^\n]{0,20}?DOF\s+\d{2}-\d{2}-\d{4}" + _FECHAS_NOTA +
    r"|(?:DOF\s+)?\d{2}-\d{2}-\d{4}" + _FECHAS_NOTA +
    r"|Reforma DOF \d{2}-\d{2}-\d{4}:[^\n]*"
    r"|Se deroga referencia[^\n]*"
    r"|Fe de erratas[^\n]*DOF[^\n]*)$",
    re.I,
)
RE_COLA_NOTA = re.compile(
    r"(?<=[.;:])\s+(?:P[áa]rrafo|Art[íi]culo|Fracci[óo]n|Inciso|Apartado|Numeral|Cap[íi]tulo|T[íi]tulo|Secci[óo]n)"
    r"[^\n]{0,40}?\s" + _VERBO_NOTA + r"\s+DOF\s+\d{2}-\d{2}-\d{4}[^\n]*$", re.I)
RE_PIE = re.compile(r"^\s*\d+\s+de\s+\d+\s*$")


def _clave_linea(t: str) -> str:
    return re.sub(r"\d+", "#", " ".join(t.split())).lower()


def extraer_parrafos(ruta_pdf: Path) -> list[str]:
    """Texto de la ley en párrafos, sin encabezados de página ni notas de reforma."""
    import fitz  # PyMuPDF

    doc = fitz.open(str(ruta_pdf))
    paginas = []
    for p in doc:
        alto = p.rect.height
        bloques = [b for b in p.get_text("blocks") if b[6] == 0]
        paginas.append((alto, bloques))
    # Los encabezados y pies se repiten en casi todas las páginas en la misma
    # franja; se aprenden por documento en vez de fijar alturas.
    repetidos = Counter()
    for alto, bloques in paginas:
        vistos = set()
        for x0, y0, x1, y1, t, *_ in bloques:
            if y1 < alto * 0.2 or y0 > alto * 0.88:
                vistos.add(_clave_linea(t))
        repetidos.update(vistos)
    umbral = max(2, int(len(paginas) * 0.5))
    lineas: list[str] = []
    for alto, bloques in paginas:
        cuerpo = []
        for x0, y0, x1, y1, t, *_ in sorted(bloques, key=lambda b: (round(b[1], 1), b[0])):
            franja = y1 < alto * 0.2 or y0 > alto * 0.88
            if franja and (repetidos[_clave_linea(t)] >= umbral or RE_PIE.match(t)):
                continue
            if RE_PIE.match(t) and y0 > alto * 0.85:
                continue
            cuerpo.append(t)
        texto = "\n".join(cuerpo)
        # Salto de página: si lo anterior cerró frase y lo nuevo abre párrafo,
        # es párrafo nuevo aunque el PDF no deje renglón en blanco.
        if lineas and texto.strip():
            prev = next((l for l in reversed(lineas) if l.strip()), "")
            primera = texto.strip().split("\n", 1)[0].strip()
            if re.search(r"[.;:]\s*$", prev) and re.match(r"(?:[A-ZÁÉÍÓÚÑ]|[IVXLC]+\.|[a-z]\)|\d+\.)", primera):
                lineas.append("")
        lineas.extend(texto.split("\n"))
    parrafos, actual = [], []
    for l in lineas:
        if l.strip():
            actual.append(l.strip())
        elif actual:
            parrafos.append(actual)
            actual = []
    if actual:
        parrafos.append(actual)
    salida = []
    for renglones in parrafos:
        # Un «párrafo» del PDF puede empezar con encabezado y su nombre en dos
        # renglones («CAPITULO I» / «De las Disposiciones Generales»): se
        # conservan los saltos para poder reconocerlos después.
        limpio = [RE_COLA_NOTA.sub("", r) for r in renglones if not RE_NOTA_REFORMA.match(r)]
        if limpio:
            salida.extend(_separar_pegados(limpio))
    return salida


def _separar_pegados(renglones: list[str]) -> list[str]:
    """Parte un bloque donde el PDF no dejó renglón en blanco.

    Vida Silvestre trae «CAPÍTULO IV / SANIDAD DE LA VIDA SILVESTRE / Artículo
    25. El control…» en un solo bloque: sin partirlo, el artículo 25 quedaba
    como nombre del capítulo y desaparecía. Un artículo sólo abre párrafo si el
    renglón anterior cerró frase o era encabezado: «…conforme al\nArtículo 25
    de la Ley…» es una cita partida por el ancho de página, no un artículo.
    """
    piezas: list[list[str]] = []
    en_encabezado = False
    for r in renglones:
        plano = " ".join(r.split())
        es_enc = bool(RE_ENCABEZADO.match(plano)) and len(plano) <= 90
        es_art = bool(RE_ARTICULO.match(plano))
        previo = piezas[-1][-1] if piezas else ""
        abre = not piezas or es_enc or (es_art and (en_encabezado or re.search(r"[.;:)]\s*$", previo)))
        if abre:
            piezas.append([r])
            en_encabezado = es_enc
        else:
            piezas[-1].append(r)
    return ["\n".join(x) for x in piezas]


# ══════════════════════════════════════════════════════════════════════
# DE PÁRRAFOS A ARTÍCULOS
# ══════════════════════════════════════════════════════════════════════
# Los multiplicativos latinos no se acaban en DECIES: Aeropuertos llega al
# «73 OCTODECIES». Todos terminan en -IES —o en -US: Discriminación usa «15
# Quintus» a «15 Novenus»—; con la lista corta se fundían nueve artículos
# distintos en un solo «Artículo 73.».
SUFIJOS = r"(?i:[A-ZÁÉÍÓÚ]{2,}[IÍ]ES|[A-ZÁÉÍÓÚ]{3,}US|QU[ÁA]TER|BIS|TER)"
# La letra del sufijo va PEGADA y en mayúscula: «69-B.», «2o.-A.-» (IVA), «32-A.-»;
# y también el número: «58-6.» (LFPCA), que es otro artículo y no el 58, y la Ñ
# (Derechos llega al «29-Ñ»).
# Con la bandera de mayúsculas encendida para todo, «1o.- La presente…» salía
# como «Artículo 1-La.»: el artículo del sujeto se leía como sufijo.
RE_ARTICULO = re.compile(
    r"^(?:ART[ÍI]CULO|Art[íi]culo)\s+(\d{1,4})\s*(?:/?o|º|°)?\.?"
    r"((?:-(?:[A-ZÑ]{1,2}|\d{1,2})(?=[\s.\-–,])|-\s[A-ZÑ](?=\.-)"
    r"|\s*[-–]?\s*" + SUFIJOS + r"(?:[\s-]*\d{1,2}(?!\d))?(?![A-Za-záéíóúñ])"
    r"|\s+[A-Z](?=\s*[.\-–]))*)"
    r"\s*(?:[.\-–:](?![,;])|$|\s+(?=[\dA-ZÁÉÍÓÚÑ(«\"]))",
)
_ORDINAL = (r"(?:PRIMER[OA]?|SEGUND[OA]|TERCER[OA]?|CUART[OA]|QUINT[OA]|SEXT[OA]|S[ÉE]PTIM[OA]|OCTAV[OA]"
            r"|NOVEN[OA]|D[ÉE]CIM[OA]|UND[ÉE]CIM[OA]|DUOD[ÉE]CIM[OA]|VIG[ÉE]SIM[OA]|TRIG[ÉE]SIM[OA]"
            r"|[ÚU]NIC[OA]|PRELIMINAR|FINAL(?:ES)?)")
# Sólo con numeral de verdad: «Sección Mexicana de…» o «Título de concesión…» a
# principio de párrafo no son encabezados.
RE_ENCABEZADO = re.compile(
    r"^(LIBRO|T[ÍI]TULO|CAP[ÍI]TULO|SECCI[ÓO]N)\s+"
    r"((?:[IVXLCDM]+|\d+|" + _ORDINAL + r"(?:\s+" + _ORDINAL + r")?|[A-Z])"
    r"(?:\s+(?:BIS|TER|[A-Z]))?)(?![A-Za-záéíóúñ])",
    re.I,
)
RE_TRANSITORIOS = re.compile(
    r"^(?:ART[ÍI]CULOS?\s+)?(?:TRANSITORIOS?|DISPOSICIONES\s+TRANSITORIAS)\s*[.:]?$", re.I)
RE_FIN_ORIGINAL = re.compile(
    r"^(?:ART[ÍI]CULOS\s+TRANSITORIOS\s+DE\s+DECRETOS?\s+DE\s+REFORMA"
    r"|DECRETO\s+(?:por\s+el\s+que|que|mediante)\b)", re.I)


def _designacion(m: re.Match) -> tuple[int, str]:
    """(71, "-B") para «Artículo 71-B.»; (32, " Bis") para «ARTÍCULO 32 BIS.-»."""
    num = int(m.group(1))
    suf = " ".join((m.group(2) or "").replace("–", "-").split())
    partes = []
    for m_suf in re.finditer(r"(-?)\s*(?:(" + SUFIJOS + r"(?:[\s-]*\d{1,2})?)(?![A-Za-záéíóúñ])"
                             r"|([A-ZÑ]{1,2}|\d{1,2})(?![A-Za-záéíóúñ\d]))", suf):
        guion, palabra, letra = m_suf.groups()
        if palabra:
            palabra = re.sub(r"[\s-]+", " ", palabra).strip()
            partes.append(("-" if guion else " ") + palabra[:1].upper() + palabra[1:].lower())
        elif letra:
            partes.append(("-" if guion else " ") + letra)
    return num, "".join(partes)


def partir_largo(texto: str) -> list[str]:
    if len(texto) <= MAX_CHARS:
        return [texto]
    partes, ini = [], 0
    while ini < len(texto):
        fin = ini + MAX_CHARS
        if fin >= len(texto):
            partes.append(texto[ini:])
            break
        corte = texto.rfind("\n", ini + MAX_CHARS // 2, fin)
        if corte == -1:
            corte = texto.rfind(". ", ini + MAX_CHARS // 2, fin)
        corte = fin if corte == -1 else corte + 1
        partes.append(texto[ini:corte])
        ini = max(corte - SOLAPE, ini + 1)
    return [p.strip() for p in partes if p.strip()]


def trocear(parrafos: list[str]) -> list[dict]:
    """Preámbulo, artículos con su jerarquía, y los transitorios de la ley original."""
    nivel = {"libro": ("", ""), "titulo": ("", ""), "capitulo": ("", ""), "seccion": ("", "")}
    menores = {"libro": ("titulo", "capitulo", "seccion"), "titulo": ("capitulo", "seccion"),
               "capitulo": ("seccion",), "seccion": ()}
    piezas: list[dict] = []
    preambulo: list[str] = []
    transitorios: list[str] = []
    zona = "preambulo"
    actual = None
    i = 0
    while i < len(parrafos):
        p = parrafos[i]
        plano = " ".join(p.split())
        if zona == "transitorios":
            if RE_FIN_ORIGINAL.match(plano):
                break
            transitorios.append(" ".join(p.split("\n")))
            i += 1
            continue
        if RE_TRANSITORIOS.match(plano) and piezas:
            zona = "transitorios"
            actual = None
            i += 1
            continue
        m_enc = RE_ENCABEZADO.match(plano)
        if m_enc and len(plano) < 220 and not RE_ARTICULO.match(plano):
            renglones = p.split("\n")
            cabeza = renglones[0].strip()
            nombre = " ".join(r.strip() for r in renglones[1:]).strip()
            # El nombre a veces viene en el párrafo siguiente, corto y sin punto final.
            if not nombre and i + 1 < len(parrafos):
                sig = " ".join(parrafos[i + 1].split())
                if (len(sig) < 160 and not RE_ARTICULO.match(sig) and not RE_ENCABEZADO.match(sig)
                        and not RE_TRANSITORIOS.match(sig) and not sig.endswith((".", ";", ":"))):
                    nombre = sig
                    i += 1
            clase = {"libro": "libro", "titulo": "titulo", "título": "titulo", "capitulo": "capitulo",
                     "capítulo": "capitulo", "seccion": "seccion", "sección": "seccion"}[m_enc.group(1).lower()]
            nivel[clase] = (cabeza, nombre)
            for menor in menores[clase]:
                nivel[menor] = ("", "")
            zona = "cuerpo"
            actual = None
            i += 1
            continue
        m_art = RE_ARTICULO.match(plano)
        if m_art:
            num, suf = _designacion(m_art)
            zona = "cuerpo"
            actual = {"num": num, "suf": suf, "parrafos": [" ".join(p.split("\n"))], **{k: v for k, v in nivel.items()}}
            piezas.append(actual)
            i += 1
            continue
        if zona == "preambulo":
            preambulo.append(" ".join(p.split("\n")))
        elif actual is not None:
            actual["parrafos"].append(" ".join(p.split("\n")))
        i += 1

    trozos = []
    pos = 0

    def campos(nv):
        titulo_c, titulo_n = nv["titulo"] if nv["titulo"][0] else nv["libro"]
        cap_c, cap_n = nv["capitulo"]
        sec_c, sec_n = nv["seccion"]
        return {"titulo": titulo_c,
                "subtitulo": titulo_n,
                "capitulo": f"{cap_c}\n\n{cap_n}" if cap_c and cap_n else cap_c,
                "seccion": f"{sec_c}\n\n{sec_n}" if sec_c and sec_n else sec_c}

    vacio = {"titulo": "", "subtitulo": "", "capitulo": "", "seccion": ""}
    if preambulo:
        for k, sub in enumerate(partir_largo("\n".join(preambulo))):
            trozos.append({"texto_raw": sub, "ref": "Preámbulo", "num": None, "chunk_index": k,
                           "es_transitorio": False, "pos": pos, **vacio})
            pos += 1
    for a in piezas:
        cuerpo = "\n".join(x for x in a["parrafos"] if x.strip())
        if len(cuerpo) < MIN_ARTICULO:
            continue
        ref = f"Artículo {a['num']}{a['suf']}."
        for k, sub in enumerate(partir_largo(cuerpo)):
            trozos.append({"texto_raw": sub, "ref": ref, "num": a["num"], "chunk_index": k,
                           "es_transitorio": False, "pos": pos, **campos(a)})
            pos += 1
    if transitorios:
        for k, sub in enumerate(partir_largo("TRANSITORIOS\n" + "\n".join(transitorios))):
            trozos.append({"texto_raw": sub, "ref": "Transitorios", "num": None, "chunk_index": k,
                           "es_transitorio": True, "pos": pos, **vacio})
            pos += 1
    return trozos


def payload_de(t: dict, ley: str, materia: str, url_pdf: str) -> dict:
    cabecera = " | ".join([ley] + [t[k] for k in ("titulo", "capitulo", "seccion", "subtitulo") if t[k]])
    jerarquia = " > ".join([ley] + [t[k] for k in ("titulo", "capitulo", "seccion") if t[k]] + [t["ref"]])
    pl = {
        "texto": f"[{cabecera}]\n{t['texto_raw']}",
        "texto_raw": t["texto_raw"],
        "ref": t["ref"],
        "ley": ley,
        "titulo": t["titulo"],
        "capitulo": t["capitulo"],
        "seccion": t["seccion"],
        "subtitulo": t["subtitulo"],
        "jerarquia": jerarquia,
        "materia": materia,
        "entidad": "FEDERAL",
        "chunk_index": t["chunk_index"],
        "es_transitorio": t["es_transitorio"],
        "source_type": "ley_federal",
        "ingestion_version": "v2_terminator",
        "cuerpo_legal_oficial": ley,
        "url_pdf": url_pdf,
        "pdf": url_pdf,
        "ingesta": MARCA,
    }
    if t["num"] is not None:
        pl["articulo_num"] = t["num"]
    return pl


def id_de(ley: str, t: dict) -> str:
    return str(uuid.UUID(hashlib.md5(f"{MARCA}|{ley}|{t['ref']}|{t['chunk_index']}|{t['pos']}".encode()).hexdigest()))


# ══════════════════════════════════════════════════════════════════════
# PROGRAMA
# ══════════════════════════════════════════════════════════════════════
def _env(ruta: Path) -> dict:
    env = {}
    for l in ruta.read_text().splitlines():
        if "=" in l and not l.startswith("#"):
            k, v = l.split("=", 1)
            env[k.strip()] = v.strip().strip('"')
    return env


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogo", default="scripts/leyes_federales_catalogo.json",
                    help="abrev → nombre, materia, pdf (ver el JSON)")
    ap.add_argument("--pdfs", required=True, help="carpeta con <abrev>.pdf de LeyesBiblio")
    ap.add_argument("--env", default=str(Path(__file__).resolve().parents[1] / ".env"))
    ap.add_argument("--abrev", nargs="*", help="sólo estas leyes")
    ap.add_argument("--seco", action="store_true", help="trocea y cuenta; no llama a ninguna API")
    ap.add_argument("--revertir", action="store_true")
    ap.add_argument("--salida", default="ingesta_federales_informe.json")
    a = ap.parse_args()

    catalogo = json.loads(Path(a.catalogo).read_text(encoding="utf-8"))
    elegidas = [c for c in catalogo if not a.abrev or c["abrev"] in a.abrev]
    if a.abrev and len(elegidas) != len(set(a.abrev)):
        faltan = set(a.abrev) - {c["abrev"] for c in elegidas}
        print("ABORTA: no están en el catálogo:", sorted(faltan))
        return 1

    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    informe = []
    for c in elegidas:
        ruta = Path(a.pdfs) / f"{c['abrev']}.pdf"
        trozos = trocear(extraer_parrafos(ruta))
        pls = [payload_de(t, c["ley"], c["materia"], c["url_pdf"]) for t in trozos]
        nums = sorted({pl["articulo_num"] for pl in pls if "articulo_num" in pl})
        tokens = sum(len(enc.encode(pl["texto"])) for pl in pls)
        informe.append({"abrev": c["abrev"], "ley": c["ley"], "trozos": len(pls), "articulos": len(nums),
                        "ultimo": nums[-1] if nums else None,
                        "huecos": [n for n in range(1, (nums[-1] if nums else 0) + 1) if n not in set(nums)][:30],
                        "tokens": tokens, "c": c, "pls": pls, "trozos_raw": trozos})
    if a.seco:
        tot_t = sum(x["tokens"] for x in informe)
        for x in informe:
            print(f"{x['abrev']:18s} {x['trozos']:5d} trozos {x['articulos']:5d} arts (últ. {x['ultimo']}) "
                  f"{x['tokens']:8d} tok  huecos {len(x['huecos'])}  {x['ley'][:70]}")
        print(f"TOTAL {len(informe)} leyes, {sum(x['trozos'] for x in informe)} trozos, {tot_t} tokens")
        Path(a.salida).write_text(json.dumps([{k: v for k, v in x.items() if k not in ('pls', 'trozos_raw', 'c')}
                                              for x in informe], ensure_ascii=False, indent=1))
        return 0

    from openai import OpenAI
    from qdrant_client import QdrantClient
    from qdrant_client.models import (FieldCondition, Filter, MatchValue, PointIdsList, PointStruct,
                                      SparseVector)
    env = _env(Path(a.env))
    q = QdrantClient(url=env["QDRANT_URL"], api_key=env["QDRANT_API_KEY"], timeout=120)
    carpeta = Path("respaldos_ingesta_federales")
    carpeta.mkdir(exist_ok=True)

    def reintentar(fn, *x, **kw):
        for i in range(5):
            try:
                return fn(*x, **kw)
            except Exception:
                if i == 4:
                    raise
                time.sleep(2 * (i + 1))

    def contar(ley=None):
        f = Filter(must=[FieldCondition(key="ley", match=MatchValue(value=ley))]) if ley else None
        return reintentar(q.count, COLECCION, count_filter=f, exact=True).count

    if a.revertir:
        for x in informe:
            f_ids = carpeta / f"{x['abrev']}.ids.json"
            if not f_ids.exists():
                print(f"{x['abrev']}: sin ids guardados; nada que revertir")
                continue
            ids = json.loads(f_ids.read_text())
            for j in range(0, len(ids), 100):
                reintentar(q.delete, COLECCION, points_selector=PointIdsList(points=ids[j:j + 100]), wait=True)
                time.sleep(PAUSA)
            print(f"REVERTIDO {x['abrev']}: {len(ids)} puntos borrados; quedan {contar(x['ley'])} con ese nombre")
        return 0

    oa = OpenAI(api_key=env["OPENAI_API_KEY"])
    antes_total = contar()
    resultados = []
    for x in informe:
        ley = x["ley"]
        if contar(ley):
            print(f"ABORTA {x['abrev']}: «{ley}» ya tiene puntos en la colección")
            return 1
        ids, gastados = [], 0
        for ini in range(0, len(x["pls"]), LOTE_EMBED):
            lote_pl = x["pls"][ini:ini + LOTE_EMBED]
            lote_t = x["trozos_raw"][ini:ini + LOTE_EMBED]
            r = reintentar(oa.embeddings.create, model=MODELO_EMBED, input=[pl["texto"] for pl in lote_pl])
            gastados += r.usage.total_tokens
            puntos = []
            for t, pl, d in zip(lote_t, lote_pl, r.data):
                idx, val = disperso(pl["texto"])
                pid = id_de(ley, t)
                puntos.append(PointStruct(id=pid, payload=pl, vector={
                    "dense": d.embedding, "sparse": SparseVector(indices=idx, values=val)}))
                ids.append(pid)
            for j in range(0, len(puntos), LOTE_UPSERT):
                reintentar(q.upsert, COLECCION, points=puntos[j:j + LOTE_UPSERT], wait=True)
                time.sleep(PAUSA)
            (carpeta / f"{x['abrev']}.ids.json").write_text(json.dumps(ids))  # revertir sirve a medias
        if len(set(ids)) != len(ids):
            print(f"ABORTA {x['abrev']}: ids repetidos; revertir")
            return 1
        for _ in range(60):
            if contar(ley) >= len(ids):
                break
            time.sleep(5)
        en_qdrant = contar(ley)
        resultados.append({"abrev": x["abrev"], "ley": ley, "puntos": len(ids), "en_qdrant": en_qdrant,
                           "articulos": x["articulos"], "tokens_embebidos": gastados,
                           "costo_usd": round(gastados * 0.02 / 1e6, 5)})
        print(json.dumps(resultados[-1], ensure_ascii=False))
    despues_total = contar()
    nuevos = sum(r["puntos"] for r in resultados)
    resumen = {"antes": antes_total, "despues": despues_total, "nuevos": nuevos,
               "ninguno_perdido": despues_total == antes_total + nuevos, "leyes": resultados,
               "cuando": time.strftime("%Y-%m-%d %H:%M")}
    Path(a.salida).write_text(json.dumps(resumen, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in resumen.items() if k != "leyes"}, ensure_ascii=False))
    return 0 if resumen["ninguno_perdido"] else 2


if __name__ == "__main__":
    sys.exit(main())
