#!/usr/bin/env python3
"""
Ingesta de la biblioteca doctrinal → colección `doctrina` en Qdrant.

EL DISEÑO QUE PROTEGE DERECHOS DE AUTOR (decisión de David, 7-ago-2026)
-----------------------------------------------------------------------
Los vectores y el texto viven en Qdrant SOLO para que el motor recupere y el
modelo entienda. Al usuario jamás se le sirve la obra: sólo cita breve,
referencia completa (autor, obra, año, PÁGINA) y enlace al PDF en su fuente.
Es el derecho de cita del art. 148 fr. I de la LFDA, hecho arquitectura.

LA PÁGINA ES EL DATO SAGRADO
----------------------------
Sin página no hay cita profesional. Por eso el troceado es POR PÁGINA del PDF
—nunca por caracteres corridos— y cada fragmento guarda `pagina_pdf` (dentro
de su archivo) y `pagina_impresa` (la del libro, detectada del propio texto),
que es la que un abogado escribe en su escrito.

EL FANTASMA DEL «ART. 19», EXORCIZADO DE ORIGEN
-----------------------------------------------
El troceado ciego de la CPEUM metió la lista de diputados de 1917 bajo
«Art. 19» y estuvo sirviéndose como norma. Aquí: portadas, índices,
bibliografías y colofones se etiquetan `subtipo` aparte y NUNCA se citan como
doctrina. Las páginas con OCR pobre se descartan y se cuentan.

FRENOS DE PRODUCCIÓN (los de siempre)
-------------------------------------
En seco por omisión. Lotes con pausa y wait=false. Vigilante de RAM a 3.20
GiB. IDs deterministas CON posición (la lección de Oaxaca: sin ella se
pierden fragmentos en silencio). Colección con vectores y payload EN DISCO:
la doctrina tolera 50 ms más; la RAM del nodo es el recurso escaso.

USO
---
    python ingestar_doctrina.py                       # en seco, todas las «listo»
    python ingestar_doctrina.py --obra fix-zamudio-ensayos-amparo
    python ingestar_doctrina.py --obra fix-zamudio-ensayos-amparo --escribir
    python ingestar_doctrina.py --escribir
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request
import uuid
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
ENV = RAIZ / ".env"
CATALOGO = Path(__file__).resolve().parent / "catalogo_doctrina.json"
CACHE = RAIZ / "_tools" / "doctrina_pdfs"     # _tools/ está en .gitignore

COLECCION = "doctrina"
MODELO_EMBED = "text-embedding-3-small"
DIM = 1536

LOTE_EMBED = 64
LOTE_UPSERT = 20
PAUSA_UPSERT = 0.35
PAUSA_DESCARGA = 0.6          # cortesía con el servidor de la UNAM
TECHO_RAM_GIB = 3.20
MIN_CARACTERES_PAGINA = 180   # menos que esto = portada, lámina u OCR muerto
MAX_CARACTERES_TROZO = 2400   # una página larga se parte en dos

RE_PAGINA_IMPRESA = re.compile(r'^\s*(\d{1,4})\s*$')
RE_PRELIMINAR = re.compile(
    r'(´?INDICE|ÍNDICE|CONTENIDO|PRESENTACI[ÓO]N|PR[ÓO]LOGO|ADVERTENCIA'
    r'|ABREVIATURAS|PORTADA|PRIMERA EDICI[ÓO]N|DERECHOS RESERVADOS|ISBN'
    r'|Formaci[óo]n en computadora|se termin[óo] de imprimir)', re.I)
RE_BIBLIOGRAFIA = re.compile(r'^\s*(BIBLIOGRAF[ÍI]A|FUENTES CONSULTADAS|HEMEROGRAF[ÍI]A)', re.I | re.M)


# ── infraestructura ──────────────────────────────────────────────────────
def cargar_env() -> dict:
    datos = {}
    for linea in ENV.read_text(encoding="utf-8", errors="ignore").splitlines():
        linea = linea.strip()
        if linea and not linea.startswith("#") and "=" in linea:
            c, v = linea.split("=", 1)
            datos[c] = v.strip().strip('"').strip("'")
    return datos


def pedir(url: str, cuerpo=None, cabeceras=None, metodo=None, intentos=4, timeout=120):
    d = json.dumps(cuerpo).encode() if isinstance(cuerpo, (dict, list)) else cuerpo
    r = urllib.request.Request(url, data=d, headers=cabeceras or {}, method=metodo)
    ultimo = None
    for i in range(intentos):
        try:
            with urllib.request.urlopen(r, timeout=timeout) as resp:
                ct = resp.headers.get("Content-Type", "")
                bruto = resp.read()
                return json.loads(bruto) if "json" in ct else bruto
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            ultimo = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"{url}: {ultimo}")


class Qdrant:
    def __init__(self, url, api):
        self.u, self.k = url.rstrip("/"), api

    def _p(self, ruta, cuerpo=None, metodo=None):
        return pedir(self.u + ruta, cuerpo,
                     {"api-key": self.k, "Content-Type": "application/json"}, metodo)

    def ram_gib(self):
        m = self._p("/telemetry?details_level=1")["result"]["memory"]
        return m["resident_bytes"] / 2 ** 30

    def existe(self):
        try:
            self._p(f"/collections/{COLECCION}")
            return True
        except Exception:
            return False

    def crear(self):
        """Vectores Y payload en disco: cero presión sobre la RAM del nodo."""
        self._p(f"/collections/{COLECCION}", {
            "vectors": {"dense": {"size": DIM, "distance": "Cosine", "on_disk": True}},
            "on_disk_payload": True,
            "hnsw_config": {"on_disk": True},
        }, metodo="PUT")
        for campo, tipo in (("autor", "keyword"), ("obra", "keyword"),
                            ("materia", "keyword"), ("subtipo", "keyword")):
            self._p(f"/collections/{COLECCION}/index?wait=true",
                    {"field_name": campo,
                     "field_schema": {"type": tipo, "on_disk": True}}, metodo="PUT")

    def upsert(self, puntos):
        self._p(f"/collections/{COLECCION}/points?wait=false", {"points": puntos}, metodo="PUT")

    def cuenta(self):
        return self._p(f"/collections/{COLECCION}")["result"].get("points_count") or 0


def embeber(claves: dict, textos: list) -> list:
    cuerpo = {"model": MODELO_EMBED, "input": textos}
    r = pedir("https://api.openai.com/v1/embeddings", cuerpo,
              {"Authorization": f"Bearer {claves['OPENAI_API_KEY']}",
               "Content-Type": "application/json"})
    return [d["embedding"] for d in r["data"]]


# ── obtención de PDFs ────────────────────────────────────────────────────
def _existe_pdf(url: str) -> bool:
    try:
        r = urllib.request.Request(url, method="HEAD",
                                   headers={"User-Agent": "Mozilla/5.0 (Iurexia biblioteca)"})
        with urllib.request.urlopen(r, timeout=30) as resp:
            return resp.status == 200
    except Exception:
        return False


def capitulos_bjv(url_detalle: str) -> list[str]:
    """Los capítulos del libro, con dos caminos.

    El bueno: la página `detalle-libro` los lista. El de respaldo: la BJV
    redirige a su portada cuando el slug no es exacto —los slugs cambian con
    las reediciones—, así que si la página no da PDFs se sondea directamente
    `archivos.juridicas.unam.mx/www/bjv/libros/{serie}/{id}/{n}.pdf`: el id es
    estable aunque el slug no lo sea, y los capítulos son consecutivos.
    """
    html = pedir(url_detalle, cabeceras={"User-Agent": "Mozilla/5.0 (Iurexia biblioteca)"}).decode("utf-8", "ignore")
    urls = re.findall(r'https?://archivos\.juridicas\.unam\.mx/www/bjv/libros/\d+/\d+/\d+\.pdf', html)
    if urls:
        return sorted(set(urls), key=lambda u: int(re.search(r'/(\d+)\.pdf$', u).group(1)))

    m = re.search(r'detalle-libro/(\d+)', url_detalle)
    if not m:
        return []
    libro = m.group(1)
    for serie in range(1, 16):
        base = f"https://archivos.juridicas.unam.mx/www/bjv/libros/{serie}/{libro}"
        if not _existe_pdf(f"{base}/1.pdf"):
            continue
        caps, fallos, n = [], 0, 1
        while fallos < 2 and n < 120:
            url = f"{base}/{n}.pdf"
            if _existe_pdf(url):
                caps.append(url); fallos = 0
            else:
                fallos += 1
            n += 1
            time.sleep(0.2)
        print(f"      (slug caducado: {len(caps)} capítulos hallados sondeando serie {serie})")
        return caps
    return []


def descargar(obra: dict) -> list[Path]:
    destino = CACHE / obra["clave"]
    destino.mkdir(parents=True, exist_ok=True)
    rutas = []
    if obra.get("bjv_detalle"):
        caps = capitulos_bjv(obra["bjv_detalle"])
        if not caps:
            print(f"      ⚠️ la página BJV no lista PDFs: {obra['bjv_detalle']}")
            return []
        for url in caps:
            n = re.search(r'/(\d+)\.pdf$', url).group(1)
            ruta = destino / f"{int(n):03d}.pdf"
            if not ruta.exists() or ruta.stat().st_size < 1024:
                datos = pedir(url, cabeceras={"User-Agent": "Mozilla/5.0 (Iurexia biblioteca)"})
                if not datos[:5].startswith(b"%PDF"):
                    print(f"      ⚠️ no es PDF: {url}")
                    continue
                ruta.write_bytes(datos)
                time.sleep(PAUSA_DESCARGA)
            rutas.append((ruta, url))
    elif obra.get("pdfs"):
        for i, url in enumerate(obra["pdfs"], 1):
            ruta = destino / f"{i:03d}.pdf"
            if not ruta.exists() or ruta.stat().st_size < 1024:
                datos = pedir(url, cabeceras={"User-Agent": "Mozilla/5.0 (Iurexia biblioteca)"})
                if not datos[:5].startswith(b"%PDF"):
                    continue
                ruta.write_bytes(datos)
                time.sleep(PAUSA_DESCARGA)
            rutas.append((ruta, url))
    return rutas


# ── extracción y troceado ────────────────────────────────────────────────
def pagina_impresa(texto: str):
    """El número que el abogado citará: vive en el margen, arriba o abajo."""
    lineas = [l for l in texto.splitlines() if l.strip()]
    for linea in (lineas[:2] + lineas[-2:] if lineas else []):
        m = RE_PAGINA_IMPRESA.match(linea)
        if m:
            n = int(m.group(1))
            if 1 <= n <= 3000:
                return n
    return None


def clasificar_pagina(texto: str) -> str:
    if RE_BIBLIOGRAFIA.search(texto[:300]):
        return "bibliografia"
    if RE_PRELIMINAR.search(texto[:500]):
        return "preliminares"
    return "doctrina"


def trocear_obra(obra: dict, rutas: list) -> tuple[list, dict]:
    import fitz  # PyMuPDF
    trozos, stats = [], {"paginas": 0, "descartadas": 0, "preliminares": 0, "bibliografia": 0}
    for ruta, url in rutas:
        try:
            doc = fitz.open(ruta)
        except Exception as e:
            print(f"      ⚠️ PDF ilegible {ruta.name}: {e}")
            continue
        for num_pag in range(len(doc)):
            texto = doc[num_pag].get_text("text").strip()
            stats["paginas"] += 1
            if len(texto) < MIN_CARACTERES_PAGINA:
                stats["descartadas"] += 1
                continue
            subtipo = clasificar_pagina(texto)
            if subtipo != "doctrina":
                stats[subtipo] += 1
            impresa = pagina_impresa(texto)
            # una página larga se parte en dos, conservando su página
            partes = ([texto] if len(texto) <= MAX_CARACTERES_TROZO
                      else [texto[:len(texto) // 2], texto[len(texto) // 2:]])
            for pos, parte in enumerate(partes):
                trozos.append({
                    "texto": parte,
                    "subtipo": subtipo,
                    "capitulo_pdf": ruta.stem,
                    "pagina_pdf": num_pag + 1,
                    "pagina_impresa": impresa,
                    "url_oficial": url,
                    "pos": pos,
                })
        doc.close()
    return trozos, stats


def id_determinista(obra: dict, t: dict) -> str:
    # CON posición: dos trozos de la misma página no deben pisarse (Oaxaca).
    base = f"doctrina|{obra['clave']}|{t['capitulo_pdf']}|{t['pagina_pdf']}|{t['pos']}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, base))


# ── programa ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obra", help="sólo esta clave del catálogo")
    ap.add_argument("--escribir", action="store_true")
    args = ap.parse_args()

    claves = cargar_env()
    q = Qdrant(claves["QDRANT_URL"], claves["QDRANT_API_KEY"])
    catalogo = json.loads(CATALOGO.read_text(encoding="utf-8"))["obras"]

    obras = [o for o in catalogo if o.get("estado") == "listo"
             and (not args.obra or o["clave"] == args.obra)]
    modo = "ESCRITURA" if args.escribir else "EN SECO (no escribe nada)"
    print(f"\nBiblioteca doctrinal · {modo} · {len(obras)} obras\n{'─' * 74}")

    total_trozos = total_paginas = 0
    for obra in obras:
        print(f"\n■ {obra['autor']} — {obra['obra']}")
        rutas = descargar(obra)
        if not rutas:
            print("      sin PDFs: se salta")
            continue
        trozos, stats = trocear_obra(obra, rutas)
        doctrina = sum(1 for t in trozos if t["subtipo"] == "doctrina")
        print(f"      {len(rutas)} PDFs · {stats['paginas']} páginas · "
              f"{doctrina} trozos de doctrina · {stats['preliminares']} preliminares · "
              f"{stats['bibliografia']} bibliografía · {stats['descartadas']} descartadas")
        con_impresa = sum(1 for t in trozos if t["pagina_impresa"])
        print(f"      página impresa detectada en {con_impresa * 100 // max(1, len(trozos))}% de los trozos")
        total_trozos += len(trozos)
        total_paginas += stats["paginas"]

        if not args.escribir:
            continue

        if q.ram_gib() >= TECHO_RAM_GIB:
            sys.exit(f"⛔ Qdrant en {q.ram_gib():.2f} GiB: alto.")
        if not q.existe():
            q.crear()
            print("      colección `doctrina` creada (vectores y payload en disco)")

        subidos = 0
        for i in range(0, len(trozos), LOTE_EMBED):
            lote = trozos[i:i + LOTE_EMBED]
            vectores = embeber(claves, [t["texto"] for t in lote])
            puntos = []
            for t, v in zip(lote, vectores):
                puntos.append({
                    "id": id_determinista(obra, t),
                    "vector": {"dense": v},
                    "payload": {
                        "texto": t["texto"],
                        "autor": obra["autor"],
                        "obra": obra["obra"],
                        "anio": obra.get("anio"),
                        "editorial": obra.get("editorial"),
                        "materia": obra.get("materia"),
                        "licencia": obra.get("licencia"),
                        "subtipo": t["subtipo"],
                        "capitulo_pdf": t["capitulo_pdf"],
                        "pagina_pdf": t["pagina_pdf"],
                        "pagina_impresa": t["pagina_impresa"],
                        "url_oficial": t["url_oficial"],
                        "fuente": "iurexia-doctrina",
                        "ingesta": "v1_2026-08",
                    },
                })
            for j in range(0, len(puntos), LOTE_UPSERT):
                q.upsert(puntos[j:j + LOTE_UPSERT])
                time.sleep(PAUSA_UPSERT)
            subidos += len(puntos)
            if subidos % 640 < LOTE_EMBED:
                print(f"      … {subidos}/{len(trozos)}", flush=True)
        print(f"      ✔ {subidos} puntos ingestados")

    print(f"\n{'─' * 74}")
    print(f"TOTAL: {total_paginas:,} páginas → {total_trozos:,} trozos")
    if args.escribir:
        print(f"puntos en `{COLECCION}`: {q.cuenta():,} · RAM Qdrant: {q.ram_gib():.3f} GiB")
    else:
        print("Relanza con --escribir.")


if __name__ == "__main__":
    main()
