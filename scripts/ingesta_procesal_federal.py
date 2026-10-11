"""Ingesta de la NORMATIVA PROCESAL FEDERAL en la colección `procesal_federal` (11-oct-2026).

Fuente: PDF vigentes de la Cámara de Diputados (LeyesBiblio), descargados el
11-oct-2026 en IUREXIA-MAC/leyes/PROCESAL_FEDERAL_2026-10/<ABREV>.pdf.

Por cada ley de `procesal_federal.LEYES`:
  1. trocea por artículo con el mismo troceador de `ingesta_leyes_federales`;
  2. acota al rango procesal cuando la ley lo mezcla con lo sustantivo;
  3. etiqueta materia, figuras, plazos y remisiones (las mismas expresiones que
     lee la consulta: `procesal_federal.py`);
  4. ubica la página del PDF donde empieza cada artículo;
  5. sube el PDF a gs://iurexia-leyes/federal/procesal/<ABREV>.pdf (visor);
  6. embebe con text-embedding-3-small (1536, el del chat) y sube a Qdrant.

Uso:  python3 scripts/ingesta_procesal_federal.py --seco      (cuenta, no gasta)
      python3 scripts/ingesta_procesal_federal.py             (ingesta)
Ids deterministas: re-correrla reescribe los mismos puntos.
"""
import argparse
import hashlib
import json
import re
import sys
import time
import uuid
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "scripts"))
import procesal_federal as pf  # noqa: E402
import ingesta_leyes_federales as ilf  # noqa: E402

PDFS = Path("/Users/josedavidalcantarmendoza/Documents/IUREXIA-MAC/leyes/PROCESAL_FEDERAL_2026-10")
BUCKET = "iurexia-leyes"
PREFIJO_GCS = "federal/procesal"
MARCA = "procesal-2026-10"
MODELO = "text-embedding-3-small"
# Los transitorios sólo se guardan donde deciden qué rige: la entrada en vigor
# del CNPCF por entidad, el régimen de la reforma de amparo de 2025, etc.
TRANSITORIOS_UTILES = {"CNPCF", "LAmp", "CNPP", "LFT", "LGMASC"}


def env() -> dict:
    d = {}
    for p in (RAIZ.parent / "SECRETOS_IUREXIA.env", RAIZ.parent / "jurexia-api-git" / ".env", RAIZ / ".env"):
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^([A-Z_0-9]+)=(.*)$", line)
            if m:
                v = m.group(2).strip()
                if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
                    v = v[1:-1]
                d[m.group(1)] = v
    return d


def paginas_por_articulo(ruta: Path) -> dict:
    import fitz
    rx = re.compile(r"^\s*ART[IÍ]CULO\s+(\d{1,4})", re.I | re.M)
    primera = {}
    for i, pag in enumerate(fitz.open(str(ruta))):
        for m in rx.finditer(pag.get_text()):
            primera.setdefault(int(m.group(1)), i + 1)
    return primera


def ultima_reforma(ruta: Path) -> str:
    import fitz
    d = fitz.open(str(ruta))
    t = " ".join(d[i].get_text() for i in range(min(2, d.page_count)))
    m = re.search(r"[ÚU]ltima\s+[Rr]eforma\s+publicada\s+DOF\s+([\d-]+)", t) or re.search(r"DOF\s+(\d{2}-\d{2}-\d{4})", t)
    return m.group(1) if m else ""


def trozos_de(abrev: str) -> list:
    nombre, materia, rango = pf.LEYES[abrev]
    ruta = PDFS / f"{abrev}.pdf"
    trozos = ilf.trocear(ilf.extraer_parrafos(ruta))
    paginas = paginas_por_articulo(ruta)
    reforma = ultima_reforma(ruta)
    url = f"https://storage.googleapis.com/{BUCKET}/{PREFIJO_GCS}/{abrev}.pdf"
    fuera = []
    for t in trozos:
        if t["es_transitorio"]:
            if abrev not in TRANSITORIOS_UTILES:
                continue
        elif rango and not (t["num"] is not None and rango[0] <= t["num"] <= rango[1]):
            continue
        norm = pf.normalizar(t["texto_raw"])
        cab_norm = pf.normalizar(" ".join(t[k] for k in ("titulo", "capitulo", "seccion", "subtitulo") if t[k]))
        figuras = pf.figuras_de(norm + " " + cab_norm)
        if t["es_transitorio"]:
            figuras = sorted(set(figuras) | {"vigencia"})
        plazos = pf.plazos_de(norm)
        cabecera = " | ".join([nombre] + [t[k] for k in ("titulo", "capitulo", "seccion", "subtitulo") if t[k]])
        pag = paginas.get(t["num"]) if t["num"] is not None else None
        pl = {
            "texto": f"[{cabecera}]\n{t['texto_raw']}",
            "texto_raw": t["texto_raw"],
            "texto_norm": norm,
            "ref": t["ref"],
            "ley": nombre,
            "abreviatura": abrev,
            "cuerpo_legal_oficial": nombre,
            "titulo": t["titulo"], "capitulo": t["capitulo"], "seccion": t["seccion"], "subtitulo": t["subtitulo"],
            "chunk_index": t["chunk_index"],
            "es_transitorio": t["es_transitorio"],
            "materia_procesal": materia,
            "figuras": figuras,
            "plazos": plazos,
            "remisiones": pf.remisiones_de(norm, t["num"]),
            "pagina": pag,
            "pdf_url": url,
            "pdf_oficial": f"https://www.diputados.gob.mx/LeyesBiblio/pdf/{'159' if abrev == 'LOTA' else abrev}.pdf",
            "ultima_reforma": reforma,
            "entidad": "FEDERAL",
            "jurisdiccion": "federal",
            "source_type": "procesal_federal",
            "ingesta": MARCA,
        }
        if t["num"] is not None:
            pl["articulo_num"] = t["num"]
        etiqueta = pf.MATERIAS_ETIQUETA.get(materia, materia)
        para_vector = (f"[{cabecera}]\n{t['texto_raw']}\n(Materia procesal: {etiqueta}. "
                       f"Figuras: {', '.join(figuras) or 'ninguna'}"
                       + (f". Plazos: {', '.join(plazos)}" if plazos else "") + ")")
        pid = str(uuid.UUID(hashlib.md5(f"{MARCA}|{abrev}|{t['ref']}|{t['chunk_index']}|{t['pos']}".encode()).hexdigest()))
        fuera.append((pid, para_vector[:7500], pl))
    return fuera


def embeber(textos, clave):
    import requests
    for i in range(6):
        r = requests.post("https://api.openai.com/v1/embeddings", timeout=120,
                          headers={"Authorization": f"Bearer {clave}"},
                          json={"model": MODELO, "input": textos})
        if r.status_code == 200:
            d = r.json()
            return [x["embedding"] for x in d["data"]], d["usage"]["total_tokens"]
        time.sleep(3 + 5 * i)
    raise RuntimeError(f"embeddings {r.status_code}: {r.text[:200]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seco", action="store_true")
    ap.add_argument("--abrev", nargs="*")
    ap.add_argument("--sin-gcs", action="store_true")
    a = ap.parse_args()
    leyes = a.abrev or list(pf.LEYES)
    todos = []
    for ab in leyes:
        tz = trozos_de(ab)
        arts = {p["articulo_num"] for _, _, p in tz if "articulo_num" in p}
        sin_pag = sum(1 for _, _, p in tz if p.get("articulo_num") and not p.get("pagina"))
        print(f"{ab:7s} trozos {len(tz):5d} artículos {len(arts):5d} sin página {sin_pag:4d} "
              f"reforma {tz[0][2]['ultima_reforma'] if tz else '?'}", flush=True)
        todos += tz
    print("TOTAL trozos", len(todos), "caracteres", sum(len(t) for _, t, _ in todos), flush=True)
    if a.seco:
        muestra = [p for _, _, p in todos if p.get("abreviatura") == "LAmp" and p.get("articulo_num") == 17][:1]
        for p in muestra:
            print(json.dumps({k: p[k] for k in ("ref", "figuras", "plazos", "remisiones", "pagina")}, ensure_ascii=False))
        return

    e = env()
    # 1) PDF al bucket
    try:
        from google.cloud import storage
        from google.oauth2 import service_account
    except ImportError:
        storage = None
    if storage is None or "--sin-gcs" in sys.argv:
        print("PDF: se omite la subida (ya están en GCS o falta google-cloud-storage)", flush=True)
    else:
        _subir_pdfs(storage, service_account, e, leyes)
    _qdrant(e, todos)


def _subir_pdfs(storage, service_account, e, leyes):
    info = json.loads(e["GCP_SA_KEY_JSON"])
    gcs = storage.Client(credentials=service_account.Credentials.from_service_account_info(info),
                         project=info.get("project_id"))
    for ab in leyes:
        b = gcs.bucket(BUCKET).blob(f"{PREFIJO_GCS}/{ab}.pdf")
        b.cache_control = "public, max-age=86400"
        b.upload_from_filename(str(PDFS / f"{ab}.pdf"), content_type="application/pdf")
    print("PDF en GCS:", len(leyes), flush=True)


def _qdrant(e, todos):
    # 2) colección
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PayloadSchemaType, PointStruct, VectorParams
    qc = QdrantClient(url=e["QDRANT_URL"], api_key=e["QDRANT_API_KEY"], timeout=120)
    if not qc.collection_exists(pf.COLECCION):
        qc.create_collection(pf.COLECCION, vectors_config={"dense": VectorParams(
            size=1536, distance=Distance.COSINE, on_disk=True)}, on_disk_payload=True)
        for campo, tipo in (("abreviatura", PayloadSchemaType.KEYWORD), ("ley", PayloadSchemaType.KEYWORD),
                            ("materia_procesal", PayloadSchemaType.KEYWORD), ("figuras", PayloadSchemaType.KEYWORD),
                            ("articulo_num", PayloadSchemaType.INTEGER)):
            qc.create_payload_index(pf.COLECCION, campo, tipo)
        print("colección creada", flush=True)

    # 3) vectores y subida
    tokens = 0
    for i in range(0, len(todos), 96):
        lote = todos[i:i + 96]
        vecs, tk = embeber([t for _, t, _ in lote], e["OPENAI_API_KEY"])
        tokens += tk
        qc.upsert(pf.COLECCION, points=[PointStruct(id=pid, vector={"dense": v}, payload=pl)
                                        for (pid, _, pl), v in zip(lote, vecs)], wait=True)
        if (i // 96) % 10 == 0:
            print(f"  {i + len(lote)}/{len(todos)} tokens {tokens}", flush=True)
    n = qc.count(pf.COLECCION, exact=True).count
    print(f"FIN puntos {n} tokens {tokens} ≈ USD {tokens * 0.02 / 1e6:.4f}", flush=True)


if __name__ == "__main__":
    main()
