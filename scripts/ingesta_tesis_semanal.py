"""Tesis nuevas del Semanario a `jurisprudencia_nacional_v3`, con su PDF oficial (11-oct-2026).

La v3 terminaba en el registro 2032214 (29-may-2026). Esto recorre los
registros siguientes en la API pública del SJF —desde la Mac: desde Render la
bloquea Incapsula—, a una petición por segundo, y por cada tesis:

  · JSON (isSemanal=true; si no, false: lo anterior al 10-jul-2026 sólo está
    en la Gaceta mensual) → payload con los MISMOS campos de la v3;
  · PDF oficial de la Corte (`/tesis/reporte/<registro>`, el del botón
    «Descargar documento») → gs://iurexia-leyes/tesis/<3 cifras>/<registro>.pdf;
  · vectores con text-embedding-3-large: «rubro» = rubro y «texto» =
    rubro + "\\n\\n" + texto (medido contra un punto de la v3: coseno 0.9998);
  · id = uuid5(NAMESPACE_URL, "tesis:<registro>"), el de la v3.

Fases: `--bajar` (JSON + PDF a disco, reanudable) y `--subir` (GCS + Qdrant).
Un registro que no es tesis (ejecutorias y votos comparten la numeración)
responde con la página de Incapsula: se distingue de un bloqueo pidiendo una
tesis testigo que sí existe.
"""
import argparse
import html
import json
import re
import sys
import time
import urllib.request
import uuid
from pathlib import Path

DIR = Path("/Users/josedavidalcantarmendoza/Documents/IUREXIA-MAC/reingesta/tesis_2026-10")
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0 Safari/537.36"
BASE = "https://sjf2.scjn.gob.mx/services/sjftesismicroservice/api/public/tesis"
TESTIGO = 2032690
INICIO = 2032215
HASTA_FECHA = "2026-10-09 23:59"
COLECCION = "jurisprudencia_nacional_v3"
MARCA = "v3_semanario_2026-10"


def pedir(url, ref):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Referer": ref, "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.headers.get("Content-Type", ""), r.read()


def json_de(reg):
    # Hasta el 2032443 (10-jul-2026) es Gaceta mensual: se pide primero así.
    for semanal in (("false", "true") if reg <= 2032443 else ("true", "false")):
        try:
            ct, b = pedir(f"{BASE}/{reg}?isSemanal={semanal}&hostName=https://sjf2.scjn.gob.mx",
                          f"https://sjf2.scjn.gob.mx/detalle/tesis/{reg}")
            time.sleep(1.05)
            if "json" in ct:
                d = json.loads(b.decode("utf-8"))
                if d.get("ius"):
                    d["_semanal"] = semanal
                    return d
        except Exception:
            time.sleep(3)
    return None


def pdf_de(reg, semanal):
    url = (f"{BASE}/reporte/{reg}?isSemanal={semanal}&nameDocto=Tesis&hostName=https://sjf2.scjn.gob.mx"
           "&soloParrafos=false&appSource=SJFAPP2020")
    # Un solo intento: Incapsula suele pedir su reto en esta ruta y no se
    # sortea. Lo que no baje se genera de los datos oficiales (`pdf_propio`).
    try:
        ct, b = pedir(url, f"https://sjf2.scjn.gob.mx/detalle/tesis/{reg}")
        time.sleep(1.05)
        if b[:5] == b"%PDF-":
            return b
    except Exception:
        time.sleep(2)
    return None


CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def pdf_propio(d, destino):
    """El PDF de la ficha armado con los datos oficiales del SJF, cuando el
    oficial no se pudo bajar. Mismo contenido que «Detalle de Tesis»; el pie
    dice de dónde salió."""
    import subprocess
    reg = str(d["ius"])
    e = html.escape
    filas = "".join(f"<tr><td><b>{k}:</b></td><td>{e(str(v))}</td></tr>" for k, v in (
        ("Registro digital", reg), ("Instancia", d.get("instancia") or ""), ("Época", d.get("epoca") or ""),
        ("Materia(s)", d.get("materias") or ""), ("Tesis", d.get("claveTesis") or ""),
        ("Fuente", (d.get("fuente") or "") + (". " + d["volumen"] if d.get("volumen") else "")),
        ("Tipo", d.get("tipoTesis") or "")) if v)
    par = lambda t: "".join(f"<p>{e(x)}</p>" for x in limpio(t).split("\n") if x)
    doc = (f"<html><head><meta charset='utf-8'><style>body{{font:11pt Arial;margin:40px}}h3{{margin:0 0 10px}}"
           f"td{{padding:2px 8px 2px 0;vertical-align:top}}.r{{font-weight:bold;margin:16px 0}}"
           f".pie{{font-size:8.5pt;color:#555;border-top:1px solid #aaa;margin-top:24px;padding-top:6px}}</style></head>"
           f"<body><h3>Tesis</h3><table>{filas}</table><div class='r'>{par(d.get('rubro'))}</div>{par(d.get('texto'))}"
           f"<div style='margin-top:12px'>{par(d.get('precedentes'))}</div><p><i>{e(d.get('textoPublicacion') or '')}</i></p>"
           f"<div class='pie'>Semanario Judicial de la Federación (SCJN) · ficha oficial: "
           f"https://sjf2.scjn.gob.mx/detalle/tesis/{reg} · documento compuesto con los datos públicos de la ficha.</div>"
           f"</body></html>")
    fh = destino.with_suffix(".html")
    fh.write_text(doc, encoding="utf-8")
    p = subprocess.Popen([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                          f"--print-to-pdf={destino}", fh.as_uri()], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(60):
        if p.poll() is not None:
            break
        time.sleep(0.5)
    if p.poll() is None:
        p.kill()
    fh.unlink(missing_ok=True)
    return destino.exists() and destino.stat().st_size > 2000


def completar_pdfs():
    n = 0
    for fj in sorted((DIR / "json").glob("*.json")):
        fp = DIR / "pdf" / f"{fj.stem}.pdf"
        if not fp.exists() and pdf_propio(json.loads(fj.read_text()), fp):
            n += 1
    print("PDF compuestos con los datos oficiales:", n, flush=True)


def limpio(h):
    if not h:
        return ""
    t = re.sub(r"</p>\s*<p[^>]*>", "\n", h)
    t = re.sub(r"<br\s*/?>", "\n", t)
    t = re.sub(r"<[^>]+>", "", t)
    t = html.unescape(t).replace("\xa0", " ")
    return "\n".join(l.strip() for l in t.splitlines() if l.strip())


def payload_de(d):
    reg = str(d["ius"])
    jur = (d.get("ta_tj") == 1) or "urisprudencia" in (d.get("tipoTesis") or "")
    return {
        "registro": reg,
        "rubro": limpio(d.get("rubro")),
        "texto": limpio(d.get("texto")),
        "localizacion": d.get("localizacion") or "",
        "clave_tesis": d.get("claveTesis") or "",
        "epoca": d.get("epoca") or "",
        "instancia": d.get("instancia") or "",
        "materia": d.get("materias") or "",
        "tipo": "JURISPRUDENCIA" if jur else "TESIS AISLADA",
        "vincula": bool(jur),
        "fuente": d.get("fuente") or "",
        "pagina": str(d.get("pagina") or ""),
        "fecha_publicacion": d.get("fechaPublicacion") or "",
        "precedentes": limpio(d.get("precedentes")),
        "ingesta": MARCA,
        "pdf_url": f"https://storage.googleapis.com/iurexia-leyes/tesis/{reg[:3]}/{reg}.pdf",
    }


def bajar(tope_fallos):
    (DIR / "json").mkdir(parents=True, exist_ok=True)
    (DIR / "pdf").mkdir(parents=True, exist_ok=True)
    reg, seguidos, hallados = INICIO, 0, 0
    while seguidos < tope_fallos:
        fj = DIR / "json" / f"{reg}.json"
        if fj.exists():
            d = json.loads(fj.read_text())
        else:
            d = json_de(reg)
            if d is None and seguidos and seguidos % 10 == 0 and json_de(TESTIGO) is None:
                print("bloqueo de Incapsula: espero 120 s", flush=True)
                time.sleep(120)
                continue
            if d is not None:
                fj.write_text(json.dumps(d, ensure_ascii=False))
        if d is None:
            seguidos += 1
        else:
            seguidos = 0
            hallados += 1
            fp = DIR / "pdf" / f"{reg}.pdf"
            if not fp.exists():
                b = pdf_de(reg, d.get("_semanal", "true"))
                if b:
                    fp.write_bytes(b)
            if hallados % 25 == 0:
                print(reg, "tesis", hallados, d.get("fechaPublicacion"), flush=True)
        reg += 1
    print("FIN bajada: último registro visto", reg - 1, "tesis", hallados, flush=True)


def subir():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ingesta_procesal_federal import env  # mismo lector de secretos
    e = env()
    import requests
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct
    qc = QdrantClient(url=e["QDRANT_URL"], api_key=e["QDRANT_API_KEY"], timeout=120)
    filas = []
    for fj in sorted((DIR / "json").glob("*.json")):
        d = json.loads(fj.read_text())
        if (d.get("fechaPublicacion") or "9999") > HASTA_FECHA:
            continue
        pl = payload_de(d)
        if not pl["rubro"] or not pl["texto"]:
            continue
        if not (DIR / "pdf" / f"{pl['registro']}.pdf").exists():
            pl.pop("pdf_url")
        filas.append(pl)
    print("tesis a subir", len(filas), flush=True)
    tokens = 0
    for i in range(0, len(filas), 40):
        lote = filas[i:i + 40]
        entradas = [p["rubro"] for p in lote] + [p["rubro"] + "\n\n" + p["texto"] for p in lote]
        for k in range(6):
            r = requests.post("https://api.openai.com/v1/embeddings", timeout=180,
                              headers={"Authorization": f"Bearer {e['OPENAI_API_KEY']}"},
                              json={"model": "text-embedding-3-large", "input": entradas})
            if r.status_code == 200:
                break
            time.sleep(5 + 5 * k)
        r.raise_for_status()
        vec = [x["embedding"] for x in r.json()["data"]]
        tokens += r.json()["usage"]["total_tokens"]
        n = len(lote)
        qc.upsert(COLECCION, wait=True, points=[
            PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"tesis:{p['registro']}")),
                        vector={"rubro": vec[j], "texto": vec[n + j]}, payload=p)
            for j, p in enumerate(lote)])
        print(f"  {i + n}/{len(filas)} tokens {tokens}", flush=True)
    print(f"FIN subida: {len(filas)} tesis, tokens {tokens} ≈ USD {tokens * 0.13 / 1e6:.3f}", flush=True)


def subir_pdfs():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ingesta_procesal_federal import env
    from google.cloud import storage
    from google.oauth2 import service_account
    info = json.loads(env()["GCP_SA_KEY_JSON"])
    b = storage.Client(credentials=service_account.Credentials.from_service_account_info(info),
                       project=info.get("project_id")).bucket("iurexia-leyes")
    n = 0
    for fp in sorted((DIR / "pdf").glob("*.pdf")):
        reg = fp.stem
        bl = b.blob(f"tesis/{reg[:3]}/{reg}.pdf")
        bl.cache_control = "public, max-age=86400"
        bl.upload_from_filename(str(fp), content_type="application/pdf")
        n += 1
    print("PDF de tesis en GCS:", n, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bajar", action="store_true")
    ap.add_argument("--pdfs", action="store_true", help="sube los PDF a GCS (python con google-cloud-storage)")
    ap.add_argument("--subir", action="store_true", help="vectores a Qdrant (python con qdrant_client)")
    ap.add_argument("--tope-fallos", type=int, default=250)
    a = ap.parse_args()
    if a.bajar:
        bajar(a.tope_fallos)
    if a.pdfs:
        completar_pdfs()
        subir_pdfs()
    if a.subir:
        subir()
